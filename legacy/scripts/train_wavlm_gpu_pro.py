import argparse
import json
import os

import h5py
import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import DATASET_PATHS
from extract_wav2vec import extract_features as build_wavlm_cache


try:
    from tensorflow.keras import mixed_precision

    mixed_precision.set_global_policy("mixed_float16")
except Exception:
    pass


def compute_alpha_vector(y_encoded, num_classes):
    counts = np.bincount(y_encoded, minlength=num_classes).astype(np.float32)
    inv = counts.sum() / np.maximum(counts, 1.0)
    alpha = inv / inv.sum() * num_classes
    return alpha.astype(np.float32)


class CategoricalFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=None, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        if self.alpha is None:
            alpha = 1.0
        else:
            alpha = tf.cast(self.alpha, tf.float32)
            if len(alpha.shape) == 1:
                alpha = tf.reshape(alpha, (1, -1))

        focal = alpha * tf.pow(1.0 - y_pred, self.gamma)
        loss = -y_true * focal * tf.math.log(y_pred)
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha})
        return config


class SEBlock(tf.keras.layers.Layer):
    def __init__(self, channels, reduction=4, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        self.gap = tf.keras.layers.GlobalAveragePooling1D()
        self.d1 = tf.keras.layers.Dense(max(1, channels // reduction), activation="gelu")
        self.d2 = tf.keras.layers.Dense(channels, activation="sigmoid")

    def call(self, x):
        s = self.gap(x)
        s = self.d1(s)
        s = self.d2(s)
        s = tf.reshape(s, [-1, 1, self.channels])
        return x * s

    def get_config(self):
        config = super().get_config()
        config.update({"channels": self.channels, "reduction": self.reduction})
        return config


class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, dilation_rate=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.conv1 = tf.keras.layers.Conv1D(filters, kernel_size, padding="same", dilation_rate=dilation_rate)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv1D(filters, kernel_size, padding="same", dilation_rate=dilation_rate)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.se = SEBlock(filters)
        self.add = tf.keras.layers.Add()
        self.shortcut_lyr = None

    def build(self, input_shape):
        if input_shape[-1] != self.filters:
            self.shortcut_lyr = tf.keras.layers.Conv1D(self.filters, 1, padding="same")
        super().build(input_shape)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.nn.gelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.se(x)
        shortcut = inputs if self.shortcut_lyr is None else self.shortcut_lyr(inputs)
        return self.add([x, shortcut])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "dilation_rate": self.dilation_rate,
            }
        )
        return config


class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = tf.keras.layers.Dense(128, activation="tanh")
        self.output_dense = tf.keras.layers.Dense(1)

    def call(self, x):
        score = self.score_dense(x)
        score = self.output_dense(score)
        score = tf.nn.softmax(score, axis=1)
        return x * score

    def get_config(self):
        return super().get_config()


class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, total_steps, alpha=1e-6):
        super().__init__()
        self.initial_lr = initial_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.alpha = alpha

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = self.initial_lr * (step / tf.maximum(1.0, float(self.warmup_steps)))
        denom = tf.maximum(1.0, float(self.total_steps - self.warmup_steps))
        progress = (step - self.warmup_steps) / denom
        cosine = self.alpha + 0.5 * (self.initial_lr - self.alpha) * (1 + tf.math.cos(np.pi * progress))
        return tf.where(step < self.warmup_steps, warmup, cosine)

    def get_config(self):
        return {
            "initial_lr": self.initial_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "alpha": self.alpha,
        }


class WavLMSequence(tf.keras.utils.Sequence):
    def __init__(self, h5_path, indices, y_all, batch_size, num_classes, augment=False, balance=False, random_state=42):
        self.h5_path = h5_path
        self.original_indices = np.asarray(indices)
        self.y_all = np.asarray(y_all)
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.augment = augment
        self.balance = balance
        self.rng = np.random.default_rng(random_state)
        self.h5_file = None
        self.X_ds = None
        self.indices = self._build_epoch_indices()

    def _build_epoch_indices(self):
        indices = self.original_indices.copy()
        if self.balance:
            per_class = []
            max_count = 0
            for cls in range(self.num_classes):
                cls_idx = indices[self.y_all[indices] == cls]
                if len(cls_idx) == 0:
                    continue
                per_class.append(cls_idx)
                max_count = max(max_count, len(cls_idx))
            balanced = []
            for cls_idx in per_class:
                sampled = self.rng.choice(cls_idx, size=max_count, replace=True)
                balanced.append(sampled)
            indices = np.concatenate(balanced)
        self.rng.shuffle(indices)
        return indices

    def __len__(self):
        return int(np.ceil(len(self.indices) / float(self.batch_size)))

    def on_epoch_end(self):
        self.indices = self._build_epoch_indices()

    def _ensure_open(self):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")
            self.X_ds = self.h5_file["X"]

    def __getitem__(self, idx):
        self._ensure_open()
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = np.array([self.X_ds[i] for i in batch_indices], dtype=np.float32)
        batch_y_labels = self.y_all[batch_indices]

        if self.augment:
            batch_x = self._augment(batch_x)

        batch_y = tf.keras.utils.to_categorical(batch_y_labels, self.num_classes).astype(np.float32)
        return batch_x, batch_y

    def _augment(self, batch_x):
        jitter = np.random.normal(0, 0.02, batch_x.shape).astype(np.float32)
        batch_x = batch_x + jitter

        if np.random.rand() < 0.8:
            mask_size = np.random.randint(10, 28)
            for i in range(len(batch_x)):
                start = np.random.randint(0, max(1, batch_x.shape[1] - mask_size))
                batch_x[i, start:start + mask_size, :] = 0.0

        if np.random.rand() < 0.6:
            feat_mask_size = np.random.randint(24, 72)
            feat_start = np.random.randint(0, max(1, batch_x.shape[2] - feat_mask_size))
            batch_x[:, :, feat_start:feat_start + feat_mask_size] = 0.0

        if np.random.rand() < 0.6:
            shift = np.random.randint(-12, 12)
            batch_x = np.roll(batch_x, shift, axis=1)

        return batch_x.astype(np.float32)


def build_model(num_classes):
    inputs = tf.keras.layers.Input(shape=(320, 768))
    x = tf.keras.layers.Dense(256)(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.nn.gelu(x)

    x = ResidualBlock(256, dilation_rate=1)(x)
    x = ResidualBlock(256, dilation_rate=2)(x)
    x = tf.keras.layers.SpatialDropout1D(0.15)(x)

    x = ResidualBlock(384, dilation_rate=1)(x)
    x = ResidualBlock(384, dilation_rate=4)(x)
    x = tf.keras.layers.SpatialDropout1D(0.2)(x)

    attn_res = x
    x = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=48, dropout=0.1)(x, x)
    x = tf.keras.layers.Add()([x, attn_res])
    x = tf.keras.layers.LayerNormalization()(x)

    x_att = TemporalAttention()(x)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(x_att)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(x_att)
    merged = tf.keras.layers.Concatenate()([avg_pool, max_pool])

    x = tf.keras.layers.GaussianDropout(0.25)(merged)
    x = tf.keras.layers.Dense(512, activation="gelu", kernel_regularizer=tf.keras.regularizers.l2(8e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.45)(x)
    x = tf.keras.layers.Dense(256, activation="gelu", kernel_regularizer=tf.keras.regularizers.l2(5e-4))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(num_classes, dtype="float32", activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


def load_labels(h5_path):
    with h5py.File(h5_path, "r") as f:
        y_raw = f["y"][:]
    labels = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in y_raw]
    labels = np.array([x for x in labels if x != "unknown"])
    return labels


def main():
    parser = argparse.ArgumentParser(description="Train a stronger GPU WavLM classifier.")
    parser.add_argument("--datasets", default=",".join(DATASET_PATHS))
    parser.add_argument("--cache_path", default="data/cached_wavlm_seq.h5")
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--extract_batch_size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--quick_run", action="store_true")
    parser.add_argument("--output_model", default="models/emotion_wavlm_gpu_pro.keras")
    parser.add_argument("--output_encoder", default="models/label_encoder_wavlm_gpu_pro.pkl")
    args = parser.parse_args()

    dataset_paths = [p.strip() for p in args.datasets.split(",") if p.strip()]

    if args.quick_run:
        args.epochs = 2
        args.batch_size = 16

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)

    build_wavlm_cache(
        h5_path=args.cache_path,
        batch_size=args.extract_batch_size,
        dataset_paths=dataset_paths,
        rebuild=args.rebuild_cache,
    )

    with h5py.File(args.cache_path, "r") as f:
        y_raw = f["y"][:]
        labels = np.array([x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in y_raw])

    valid_indices = np.where(labels != "unknown")[0]
    labels = labels[valid_indices]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(labels)
    num_classes = len(encoder.classes_)
    joblib.dump(encoder, args.output_encoder)

    train_idx, val_idx = train_test_split(
        valid_indices,
        test_size=0.15,
        random_state=42,
        stratify=y_encoded,
    )
    y_all = np.empty(valid_indices.max() + 1, dtype=np.int32)
    y_all[valid_indices] = y_encoded

    train_labels = y_all[train_idx]
    alpha = compute_alpha_vector(train_labels, num_classes)

    train_seq = WavLMSequence(
        args.cache_path,
        train_idx,
        y_all,
        args.batch_size,
        num_classes,
        augment=True,
        balance=True,
    )
    val_seq = WavLMSequence(
        args.cache_path,
        val_idx,
        y_all,
        args.batch_size,
        num_classes,
        augment=False,
        balance=False,
    )

    model = build_model(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=CategoricalFocalLoss(gamma=2.0, alpha=alpha),
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=12 if not args.quick_run else 2,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            args.output_model,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4 if not args.quick_run else 1,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print("=" * 72)
    print("  WavLM GPU Pro Training")
    print("=" * 72)
    print(f"Datasets: {dataset_paths}")
    print(f"Train samples: {len(train_idx)} | Val samples: {len(val_idx)}")
    print(f"Classes: {list(encoder.classes_)}")
    print(f"Alpha vector: {alpha.tolist()}")

    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    best_model = tf.keras.models.load_model(
        args.output_model,
        custom_objects={
            "CategoricalFocalLoss": CategoricalFocalLoss,
            "SEBlock": SEBlock,
            "ResidualBlock": ResidualBlock,
            "TemporalAttention": TemporalAttention,
            "WarmupCosineDecay": WarmupCosineDecay,
        },
    )

    probs = best_model.predict(val_seq, verbose=0)
    y_true = y_all[val_idx]
    y_pred = np.argmax(probs, axis=1)
    acc = accuracy_score(y_true, y_pred)

    print("=" * 72)
    print(f"FINAL VAL ACCURACY: {acc * 100:.2f}%")
    print("=" * 72)
    print(classification_report(y_true, y_pred, target_names=encoder.classes_, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print(
        "[DONE] "
        + json.dumps(
            {
                "val_accuracy": float(acc),
                "best_val_accuracy": float(max(history.history.get("val_accuracy", [0.0]))),
                "classes": list(encoder.classes_),
            }
        )
    )


if __name__ == "__main__":
    main()
