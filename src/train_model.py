import argparse
import json
import os
import sys

import joblib
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import (
    DATASET_PATHS,
    ENCODER_PATH,
    MODEL_PATH_BASE,
    SCALER_PATH,
)
from src.dataset_builder import build_multi_dataset
from src.model import ResidualBlock, SEBlock, TemporalAttention, build_model


class JsonEpochLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        payload = {
            "epoch": int(epoch + 1),
            "loss": float(logs.get("loss", 0.0)),
            "val_loss": float(logs.get("val_loss", 0.0)),
            "accuracy": float(logs.get("accuracy", 0.0)),
            "val_accuracy": float(logs.get("val_accuracy", 0.0)),
        }
        print("[EPOCH] " + json.dumps(payload), flush=True)


def _ensure_dir(path):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def _prepare_labels(y, label_encoder=None):
    y = np.asarray(y)
    if np.issubdtype(y.dtype, np.integer):
        classes = np.unique(y)
        num_classes = int(classes.max()) + 1
        y_encoded = y.astype(np.int32)
        return y_encoded, label_encoder, num_classes

    if label_encoder is None:
        label_encoder = LabelEncoder()
        label_encoder.fit(y)

    y_encoded = label_encoder.transform(y).astype(np.int32)
    return y_encoded, label_encoder, len(label_encoder.classes_)


def train_emotion_model(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    *,
    label_encoder=None,
    scaler=None,
    output_path=MODEL_PATH_BASE,
    epochs=120,
    batch_size=128,
    learning_rate=3e-4,
    dropout=0.45,
    inputs_are_scaled=False,
    verbose=2,
    save_artifacts=False,
):
    X_train = np.asarray(X_train, dtype=np.float32)
    if X_val is not None:
        X_val = np.asarray(X_val, dtype=np.float32)

    y_train_encoded, label_encoder, num_classes = _prepare_labels(y_train, label_encoder)

    if y_val is None:
        X_train, X_val, y_train_encoded, y_val_encoded = train_test_split(
            X_train,
            y_train_encoded,
            test_size=0.15,
            random_state=42,
            stratify=y_train_encoded,
        )
    else:
        y_val_encoded, _, _ = _prepare_labels(y_val, label_encoder)

    if not inputs_are_scaled:
        if scaler is None:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
        else:
            X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
    elif scaler is None:
        scaler = StandardScaler().fit(X_train)

    X_train_r = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val_r = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    y_train_ohe = tf.keras.utils.to_categorical(y_train_encoded, num_classes)
    y_val_ohe = tf.keras.utils.to_categorical(y_val_encoded, num_classes)

    _ensure_dir(output_path)
    model = build_model(
        (X_train.shape[1], 1),
        num_classes,
        learning_rate=learning_rate,
        dropout=dropout,
    )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train_encoded),
        y=y_train_encoded,
    )
    class_weight_dict = {
        int(cls): float(weight) for cls, weight in zip(np.unique(y_train_encoded), class_weights)
    }

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=18,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=6,
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            output_path,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        JsonEpochLogger(),
    ]

    history = model.fit(
        X_train_r,
        y_train_ohe,
        validation_data=(X_val_r, y_val_ohe),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=verbose,
    )

    if os.path.exists(output_path):
        model = tf.keras.models.load_model(
            output_path,
            custom_objects={
                "TemporalAttention": TemporalAttention,
                "SEBlock": SEBlock,
                "ResidualBlock": ResidualBlock,
            },
        )

    if save_artifacts and label_encoder is not None:
        _ensure_dir(SCALER_PATH)
        joblib.dump(scaler, SCALER_PATH)
        joblib.dump(label_encoder, ENCODER_PATH)

    return model, history, scaler, label_encoder


def main():
    parser = argparse.ArgumentParser(description="Train the base SER classifier.")
    parser.add_argument("--datasets", default=",".join(DATASET_PATHS[:3]))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.45)
    parser.add_argument("--output", default=MODEL_PATH_BASE)
    args = parser.parse_args()

    dataset_paths = [p.strip() for p in args.datasets.split(",") if p.strip()]
    X, y = build_multi_dataset(dataset_paths)
    label_encoder = LabelEncoder()
    label_encoder.fit(y)
    y_encoded = label_encoder.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    model, history, scaler, _ = train_emotion_model(
        X_train,
        y_train,
        X_val=X_test,
        y_val=y_test,
        label_encoder=label_encoder,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dropout=args.dropout,
        save_artifacts=True,
    )

    best_val_acc = max(history.history.get("val_accuracy", [0.0]))
    print("[DONE] " + json.dumps({"best_val_accuracy": float(best_val_acc)}), flush=True)


if __name__ == "__main__":
    main()
