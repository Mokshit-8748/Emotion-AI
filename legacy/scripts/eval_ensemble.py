"""
eval_ensemble.py
Consensus Fusion of Spectrogram-CRNN and WavLM-Attention.
Goal: Hit the 90% threshold by combining visual and temporal features.
"""
import os
import sys
import numpy as np
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Ensure project root in path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import specific components
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import (
    Dense, Dropout, BatchNormalization, LSTM, Bidirectional, 
    SpatialDropout1D, Input, GaussianNoise, Layer, Concatenate, GlobalAveragePooling1D
)
from tensorflow.keras.regularizers import l2
from src.transfer_model import build_crnn_model

# ═══════════════════════════════════════════════════════════════════════
# 1. DEFINE ARCHITECTURES (FOR LOADING WEIGHTS)
# ═══════════════════════════════════════════════════════════════════════
class GlobalAttention(Layer):
    def __init__(self, **kwargs):
        super(GlobalAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attn_weight', shape=(input_shape[-1], 1), initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='attn_bias', shape=(input_shape[1], 1), initializer='zeros', trainable=True)
        super(GlobalAttention, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        return tf.keras.backend.sum(output, axis=1)

def build_wavlm_attn_model(num_classes=7):
    inputs = Input(shape=(128, 768))
    x = GaussianNoise(0.1)(inputs)
    x = SpatialDropout1D(0.4)(x)
    lstm_seq = Bidirectional(LSTM(128, return_sequences=True, unroll=True, kernel_regularizer=l2(1e-4)))(x)
    lstm_seq = BatchNormalization()(lstm_seq)
    lstm_context = Bidirectional(LSTM(64, return_sequences=True, unroll=True, kernel_regularizer=l2(1e-4)))(lstm_seq)
    lstm_context = BatchNormalization()(lstm_context)
    attn_vec = GlobalAttention()(lstm_context)
    avg_pool = GlobalAveragePooling1D()(lstm_context)
    merged = Concatenate()([attn_vec, avg_pool])
    merged = Dropout(0.5)(merged)
    fc = Dense(256, activation='relu', kernel_regularizer=l2(1e-4))(merged)
    fc = BatchNormalization()(fc)
    fc = Dropout(0.4)(fc)
    outputs = Dense(num_classes, activation="softmax")(fc)
    return Model(inputs, outputs)

# ═══════════════════════════════════════════════════════════════════════
# 2. LOAD DATA AND MODELS
# ═══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  EMOTION AI 90% ENSEMBLE EVALUATOR")
print("=" * 70)

# Load cached data
print("\nLoading data caches...")
spec_data = np.load("data/cached_spectrograms.npz", allow_pickle=True)
wavlm_data = np.load("data/cached_wavlm_seq.npz", allow_pickle=True)

X_spec, y_labels = spec_data['X'], spec_data['y']
X_wavlm = wavlm_data['X']

# Common Split (Ensures same test set indices)
indices = np.arange(len(y_labels))
_, test_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y_labels)

X_spec_test = X_spec[test_idx]
X_wavlm_test = X_wavlm[test_idx]
y_test_labels = y_labels[test_idx]

# Load Encoders
le_wavlm = joblib.load("models/label_encoder_wavlm.pkl")
le_crnn = joblib.load("models/label_encoder_crnn.pkl")

# Verify Label Mapping
print(f"Classes (WavLM): {le_wavlm.classes_}")
print(f"Classes (CRNN):  {le_crnn.classes_}")

# Load Models
print("\nLoading Models...")
wavlm_model = build_wavlm_attn_model()
wavlm_model.load_weights("models/emotion_wavlm_best.weights.h5")

crnn_model = build_crnn_model(input_shape=(128, 128, 3), num_classes=len(le_crnn.classes_))
crnn_model.load_weights("models/emotion_crnn_best.weights.h5")

# ═══════════════════════════════════════════════════════════════════════
# 3. TEST-TIME AUGMENTATION (TTA) ENGINE
# ═══════════════════════════════════════════════════════════════════════
def apply_tta_predict(model, X, mode='wavlm', passes=3):
    """
    Predicts multiple variations of the same audio to find a robust consensus.
    """
    preds_accum = []
    for i in range(passes):
        if i == 0:
            # Pass 0: Original
            p = model.predict(X, batch_size=64, verbose=0)
        else:
            # Pass 1+: Augmented Feature Space
            X_aug = X.copy()
            if mode == 'wavlm':
                # Add slight feature jitter
                noise = np.random.normal(0, 0.02, X_aug.shape)
                X_aug += noise
                # Random Temporal Masking
                for j in range(len(X_aug)):
                    start = np.random.randint(0, 110)
                    X_aug[j, start:start+15, :] = 0.0
            else:
                # CRNN Mel jitter
                noise = np.random.normal(0, 0.01, X_aug.shape)
                X_aug = np.clip(X_aug + noise, 0, 1)
            
            p = model.predict(X_aug, batch_size=64, verbose=0)
        preds_accum.append(p)
    
    return np.mean(preds_accum, axis=0)

print("\nRunning Inference with TTA (Test-Time Augmentation)...")

# Predictions with TTA (Consensus of 3 versions each)
p_wavlm = apply_tta_predict(wavlm_model, X_wavlm_test, mode='wavlm', passes=3)
p_crnn = apply_tta_predict(crnn_model, X_spec_test, mode='crnn', passes=3)

# Consensus Weighting (WavLM Focal-tuned is the primary expert)
W_WAVLM = 0.65
W_CRNN = 0.35

p_ensemble = (W_WAVLM * p_wavlm) + (W_CRNN * p_crnn)
y_pred_idx = np.argmax(p_ensemble, axis=1)
y_pred = le_wavlm.inverse_transform(y_pred_idx)


# ═══════════════════════════════════════════════════════════════════════
# 4. FINAL ACCURACY & REPORT
# ═══════════════════════════════════════════════════════════════════════
acc = accuracy_score(y_test_labels, y_pred)
print(f"\n{'#' * 70}")
print(f"  FINAL ENSEMBLE ACCURACY: {acc*100:.2f}%")
print(f"{'#' * 70}")

print("\nDetailed Performance Report:")
print(classification_report(y_test_labels, y_pred))

# Confusion Matrix for weak links
cm = confusion_matrix(y_test_labels, y_pred)
print("\nConfusion Matrix:")
print(cm)
