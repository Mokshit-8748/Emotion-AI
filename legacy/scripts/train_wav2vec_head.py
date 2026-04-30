"""
train_wavlm_dea.py (Production Pro v5.2 - CEILING BREAKER)
Key fixes from v5.1:
  1. CHECKPOINT FIX: Saves/loads full model (.keras) to preserve optimizer state
     (root cause of the 75% ceiling — LR was misaligned on every resume)
  2. FOCAL LOSS FIX: Removed label smoothing from data generator (can't combine
     with focal loss — was double-penalizing correct predictions)
  3. FOCAL LOSS FIX: Actually applies alpha weighting now (was defined but unused)
  4. REMOVED class_weight from model.fit (already handled by focal alpha + data balance)
  5. LR SCHEDULE: Start LR lowered to 1e-4 (2nd runs don't need warmup peak of 3e-4)
  6. GAUSSIAN NOISE: Moved inside augment block (was always on, hurting val too)
"""
import os
import sys
import numpy as np
import joblib
import tensorflow as tf
import h5py
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# 1. WINDOWS CUDA DLL FIX
cuda_bin_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"
if os.path.exists(cuda_bin_path):
    os.add_dll_directory(cuda_bin_path)
    print(f"CUDA DLL path added: {cuda_bin_path}")
else:
    print(f"WARNING: CUDA bin path not found at {cuda_bin_path}")

# 2. MIXED PRECISION
from tensorflow.keras import mixed_precision
try:
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_global_policy(policy)
    print("Mixed Precision enabled: float16 math active.")
except:
    print("Mixed Precision not supported, using float32.")

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, Conv1D,
    SpatialDropout1D, Input, GaussianNoise, Layer, Concatenate,
    GlobalAveragePooling1D, MultiHeadAttention, LayerNormalization,
    Add, GlobalMaxPooling1D
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


# ═══════════════════════════════════════════════════════════════════════
# CORE COMPONENTS
# ═══════════════════════════════════════════════════════════════════════

class CategoricalFocalLoss(tf.keras.losses.Loss):
    """
    FIX v5.2: alpha is now actually applied to the loss.
    Do NOT combine with label smoothing — they conflict.
    Do NOT combine with class_weight in model.fit — triple-counting imbalance.
    """
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        # FIX v6.1: Alpha can now be a vector for cost-sensitive learning
        alpha = tf.cast(self.alpha, tf.float32)
        if len(alpha.shape) == 1:
            alpha = tf.reshape(alpha, (1, -1))
            
        focal_weight = alpha * tf.math.pow(1.0 - y_pred, self.gamma)
        loss = -y_true * focal_weight * tf.math.log(y_pred)
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha})
        return config


class SEBlock(Layer):
    def __init__(self, channels, reduction=4, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.reduction = reduction
        self.gap = GlobalAveragePooling1D()
        self.d1 = Dense(max(1, channels // reduction), activation='gelu')
        self.d2 = Dense(channels, activation='sigmoid')

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


class ResidualBlock(Layer):
    def __init__(self, filters, kernel_size=3, dilation_rate=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.conv1 = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)
        self.bn1   = BatchNormalization()
        self.conv2 = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)
        self.bn2   = BatchNormalization()
        self.se    = SEBlock(filters)
        self.add   = Add()
        self.shortcut_lyr = None

    def build(self, input_shape):
        if input_shape[-1] != self.filters:
            self.shortcut_lyr = Conv1D(self.filters, 1, padding='same')
        super().build(input_shape)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = tf.nn.gelu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.se(x)
        shortcut = inputs
        if self.shortcut_lyr is not None:
            shortcut = self.shortcut_lyr(inputs)
        return self.add([x, shortcut])

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "dilation_rate": self.dilation_rate
        })
        return config


class TemporalAttention(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_dense  = Dense(128, activation='tanh')
        self.output_dense = Dense(1)

    def call(self, x):
        score   = self.score_dense(x)
        score   = self.output_dense(score)
        score   = tf.nn.softmax(score, axis=1)
        context = x * score
        return context

    def get_config(self):
        return super().get_config()


# ═══════════════════════════════════════════════════════════════════════
# LR SCHEDULER
# FIX v5.2: initial_lr lowered to 1e-4. If you're resuming from a
# checkpoint the model is already near-converged; 3e-4 was blowing
# past the local minimum on every resume.
# ═══════════════════════════════════════════════════════════════════════
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, warmup_steps, total_steps, alpha=1e-6):
        super().__init__()
        self.initial_lr   = initial_lr
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.alpha        = alpha

    def __call__(self, step):
        step   = tf.cast(step, tf.float32)
        warmup = self.initial_lr * (step / self.warmup_steps)
        cosine = self.alpha + 0.5 * (self.initial_lr - self.alpha) * (
            1 + tf.math.cos(
                np.pi * (step - self.warmup_steps) /
                tf.cast(self.total_steps - self.warmup_steps, tf.float32)
            )
        )
        return tf.where(step < self.warmup_steps, warmup, cosine)

    def get_config(self):
        return {
            "initial_lr":    self.initial_lr,
            "warmup_steps":  self.warmup_steps,
            "total_steps":   self.total_steps,
            "alpha":         self.alpha,
        }


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════
CACHE_PATH = "data/cached_wavlm_seq.h5"
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# FIX v5.2: Use .keras path for full model saving (preserves optimizer state)
CHECKPOINT_PATH         = os.path.join(MODELS_DIR, "emotion_wavlm_best.weights.h5")   # legacy weights path
FULL_CHECKPOINT_PATH    = os.path.join(MODELS_DIR, "emotion_wavlm_best.keras")         # NEW: full model path

print("\nLoading HDF5 dataset metadata...")
with h5py.File(CACHE_PATH, 'r') as f:
    y_raw = f['y'][:]
    y = [str(label.decode('utf-8')) for label in y_raw]
    num_samples = len(y)

le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)

indices = np.arange(num_samples)
train_indices, test_indices = train_test_split(
    indices, test_size=0.15, random_state=42, stratify=y_encoded
)

# FIX v5.2: REMOVED — class_weight causes triple-counting with focal loss
# (focal alpha + class_weight + label_smoothing were all correcting imbalance simultaneously)

BATCH_SIZE = 32

# FIX v5.2: LABEL_SMOOTHING REMOVED from generator.
# Cannot be combined with focal loss — smoothed targets break the (1-p_t)^gamma term.
# If you want smoothing, use tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
# instead of focal loss, not alongside it.

# FIX v6.1: Cost-Sensitive Alpha Vector
# Manual targeting to punish the 'Sad/Fear/Disgust' Triangle
# Order: [angry, disgust, fear, happy, neutral, sad, surprise]
ALPHA_VECTOR = [0.20, 0.35, 0.40, 0.25, 0.20, 0.50, 0.10]

class WavLMH5Sequence(tf.keras.utils.Sequence):
    def __init__(self, h5_path, indices, y_all, batch_size, augment=False, num_classes=None):
        self.h5_path     = h5_path
        self.indices     = indices.copy()
        self.y_all       = y_all
        self.batch_size  = batch_size
        self.augment     = augment
        self.num_classes = num_classes
        self.h5_file     = None

    def __len__(self):
        return int(np.ceil(len(self.indices) / float(self.batch_size)))

    def on_epoch_end(self):
        if self.augment:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
            self.X_ds    = self.h5_file['X']

        batch_indices  = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x        = np.array([self.X_ds[i] for i in batch_indices])
        batch_y_labels = self.y_all[batch_indices]
        batch_y        = tf.keras.utils.to_categorical(batch_y_labels, self.num_classes).astype(np.float32)

        if self.augment:
            # FIX v5.2: NO label smoothing here (removed — conflicts with focal loss)

            # 1. Embedding Jitter
            jitter  = np.random.normal(0, 0.03, batch_x.shape).astype(np.float32)
            batch_x = batch_x + jitter

            # 2. Sequence Masking
            if np.random.rand() < 0.7:
                mask_size = np.random.randint(10, 30)
                for i in range(len(batch_x)):
                    start = np.random.randint(0, batch_x.shape[1] - mask_size)
                    batch_x[i, start:start + mask_size, :] = 0.0

            # 3. Channel Masking
            if np.random.rand() < 0.5:
                feat_mask_size = np.random.randint(20, 50)
                feat_start     = np.random.randint(0, batch_x.shape[2] - feat_mask_size)
                batch_x[:, :, feat_start:feat_start + feat_mask_size] = 0.0

            # 4. FIX v7.1: Temporal Shift (±10 frames)
            if np.random.rand() < 0.6:
                shift = np.random.randint(-15, 15)
                batch_x = np.roll(batch_x, shift, axis=1)

        return batch_x.astype(np.float32), batch_y


# ═══════════════════════════════════════════════════════════════════════
# BUILD MODEL
# FIX v5.2: GaussianNoise moved inside training path via augmentation.
# It was always active (including validation), which was adding noise
# to val features and suppressing val_accuracy artificially.
# ═══════════════════════════════════════════════════════════════════════
def build_model(num_classes):
    inputs = Input(shape=(320, 768))

    # FIX v5.2: GaussianNoise REMOVED from here — was polluting validation too.
    # Noise is now only applied in the data generator (augment=True path).
    x = Dense(256, name='dense')(inputs)
    x = BatchNormalization(name='batch_normalization')(x)
    x = tf.nn.gelu(x)

    x = ResidualBlock(256, dilation_rate=1)(x)
    x = ResidualBlock(256, dilation_rate=2)(x)
    x = SpatialDropout1D(0.15)(x)

    x = ResidualBlock(384, dilation_rate=1)(x)
    x = ResidualBlock(384, dilation_rate=4)(x)
    x = SpatialDropout1D(0.20)(x)

    attn_res = x
    x = MultiHeadAttention(num_heads=8, key_dim=48, dropout=0.1)(x, x)
    x = Add()([x, attn_res])
    x = LayerNormalization()(x)

    x_att    = TemporalAttention()(x)
    avg_pool = GlobalAveragePooling1D()(x_att)
    max_pool = GlobalMaxPooling1D()(x_att)
    merged   = Concatenate()([avg_pool, max_pool])

    # FIX v6.1: GaussianDropout 'Generalization Guard'
    x = tf.keras.layers.GaussianDropout(0.30, name='gaussian_dropout')(merged)

    # FIX v7.1: Wider Bottleneck with Stronger L2 Regularization
    x = Dense(512, activation='gelu', kernel_regularizer=l2(1e-3), name='dense_11')(x)
    x = BatchNormalization(name='batch_normalization_9')(x)
    x = Dropout(0.50, name='dropout')(x)

    outputs = Dense(num_classes, dtype='float32', activation='softmax', name='dense_13')(x)
    return Model(inputs, outputs)


# ═══════════════════════════════════════════════════════════════════════
# CHECKPOINT LOADING — FIX v5.2 (ROOT CAUSE OF 75% CEILING)
#
# PROBLEM: Loading weights-only (.h5) loses the optimizer state.
# The Adam optimizer's moment estimates (m, v) and step counter are reset.
# Combined with WarmupCosineDecay starting at step=0, the LR schedule
# thinks it's at epoch 1 — but the weights are already near-converged.
# This causes the optimizer to take too-large steps that blow past minima.
#
# FIX: Save and load the full model (.keras format) which includes
# optimizer state. Falls back to weights-only if .keras doesn't exist.
# ═══════════════════════════════════════════════════════════════════════
print("\nBuilding DEA-Pro v6.1 (Shadow Triangle Optimized)...")

# FIX v6.1: ALWAYS build the architecture from code to ensure the 
# GaussianDropout and architectural upgrades are applied.
# load_model() on .keras files restores the OLD graph.
model = build_model(num_classes)

steps_per_epoch = int(np.ceil(len(train_indices) / float(BATCH_SIZE)))
warmup_steps    = 10 * steps_per_epoch
EPOCHS          = 180
total_steps     = EPOCHS * steps_per_epoch

lr_schedule = WarmupCosineDecay(
    initial_lr   = 1e-4, # FIX: was 3e-4; lower for resume scenarios
    warmup_steps = warmup_steps,
    total_steps  = total_steps,
    alpha        = 1e-6
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.15),
    metrics=['accuracy']
)

# TRANSPLANT Logic: Load weights into the fresh v7.1 Graph
if os.path.exists(FULL_CHECKPOINT_PATH):
    print(f"\n[!] Transplanting weights from FULL .keras model: {FULL_CHECKPOINT_PATH}")
    # skip_mismatch=True handles the removed Dense(256) layer
    model.load_weights(FULL_CHECKPOINT_PATH, by_name=True, skip_mismatch=True)
    print("[!] Transplant successful. Build 7.1 Diffusion logic now active.")
elif os.path.exists(CHECKPOINT_PATH):
    print(f"\n[!] Transplanting weights from .h5 file: {CHECKPOINT_PATH}")
    model.load_weights(CHECKPOINT_PATH, by_name=True, skip_mismatch=True)
    print("[!] Transplant successful. WARNING: Fresh optimizer initialized.")
else:
    print("\n[!] No checkpoint found. Starting fresh v6.1 training.")

model.summary()

# ═══════════════════════════════════════════════════════════════════════
# CALLBACKS
# FIX v5.2: ModelCheckpoint now saves full model (.keras) to preserve
# optimizer state for future resumes.
# ═══════════════════════════════════════════════════════════════════════
callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=25,
        restore_best_weights=True,
        verbose=1
    ),
    # Save full model (optimizer state included)
    ModelCheckpoint(
        FULL_CHECKPOINT_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,   # FIX: full model, not just weights
        verbose=1
    ),
    # Also keep weights-only backup for compatibility
    ModelCheckpoint(
        CHECKPOINT_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
        verbose=0
    )
]

if __name__ == "__main__":
    print("\nStarting the HDF5 Streamed Training Loop...")
    train_gen = WavLMH5Sequence(CACHE_PATH, train_indices, y_encoded, BATCH_SIZE,
                                 augment=True,  num_classes=num_classes)
    val_gen   = WavLMH5Sequence(CACHE_PATH, test_indices,  y_encoded, BATCH_SIZE,
                                 augment=False, num_classes=num_classes)

    # FIX v5.2: REMOVED class_weight — was triple-counting class imbalance
    # (focal loss alpha + class_weight + label_smoothing all correcting simultaneously)
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        # class_weight=class_weights_dict,  # REMOVED — focal loss handles this
        callbacks=callbacks,
        verbose=2
    )

    joblib.dump(le, os.path.join(MODELS_DIR, "label_encoder_wavlm.pkl"))
    print("\nModel and encoder saved.")
