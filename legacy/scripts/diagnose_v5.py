import os
import numpy as np
import tensorflow as tf
import h5py
import joblib
from sklearn.metrics import classification_report, confusion_matrix

# Load Custom Components
from train_wav2vec_head import (
    CategoricalFocalLoss, SEBlock, ResidualBlock, 
    TemporalAttention, WarmupCosineDecay, WavLMH5Sequence
)

# Paths
MODELS_DIR  = "models"
MODEL_PATH  = os.path.join(MODELS_DIR, "emotion_wavlm_best.keras")
ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder_wavlm.pkl")
CACHE_PATH  = "data/cached_wavlm_seq.h5"

print(f"\n[!] Loading Build 5.2 Model: {MODEL_PATH}")
custom_objects = {
    'CategoricalFocalLoss': CategoricalFocalLoss,
    'SEBlock': SEBlock,
    'ResidualBlock': ResidualBlock,
    'TemporalAttention': TemporalAttention,
    'WarmupCosineDecay': WarmupCosineDecay,
}

model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
le = joblib.load(ENCODER_PATH)

print("\n[!] Loading Test Dataset...")
with h5py.File(CACHE_PATH, 'r') as f:
    y_raw = f['y'][:]
    y = [str(label.decode('utf-8')) for label in y_raw]
    y_encoded = le.transform(y)
    num_samples = len(y)

# Use the same split logic as training (random_state 42)
from sklearn.model_selection import train_test_split
indices = np.arange(num_samples)
_, test_indices = train_test_split(
    indices, test_size=0.15, random_state=42, stratify=y_encoded
)

print(f"[!] Evaluation on {len(test_indices)} test samples...")
test_gen = WavLMH5Sequence(CACHE_PATH, test_indices, y_encoded, batch_size=32, augment=False, num_classes=len(le.classes_))

# Predict
probabilities = model.predict(test_gen, verbose=1)
y_pred        = np.argmax(probabilities, axis=1)
y_true        = y_encoded[test_indices]

# Report
print("\n" + "="*60)
print("             BUILD 5.2 - CONFUSION ANALYSIS")
print("="*60)
print(classification_report(y_true, y_pred, target_names=le.classes_))

print("\nConfusion Matrix (Rows=True, Cols=Pred):")
cm = confusion_matrix(y_true, y_pred)
header = "        " + "  ".join([c[:5] for c in le.classes_])
print(header)
for i, row in enumerate(cm):
    label = le.classes_[i][:7].ljust(8)
    print(f"{label} {row}")

print("\n[!] Diagnosis Complete.")
