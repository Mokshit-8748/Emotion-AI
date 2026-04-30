"""
╔══════════════════════════════════════════════════════════════════╗
║              EMOTION AI — Central Configuration                  ║
║  All paths, constants, and settings in one place.                ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os

# ── Project root ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# ── Dataset paths ────────────────────────────────────────────────────────────
DEFAULT_DATASET_NAMES = ["RAVDESS", "CREMA-D", "TESS", "EMO-DB", "SAVEE", "IEMOCAP"]


def _discover_project_datasets():
    discovered = []
    for dataset_name in DEFAULT_DATASET_NAMES:
        candidate = os.path.join(DATA_DIR, dataset_name)
        if os.path.isdir(candidate):
            discovered.append(candidate)
    return discovered


DATASET_PATHS = _discover_project_datasets()

# ── Model paths ──────────────────────────────────────────────────────────────
MODELS_DIR          = os.path.join(PROJECT_ROOT, "models")
LEGACY_MODELS_DIR   = os.path.join(MODELS_DIR, "legacy")
MODEL_PATH_BASE     = os.path.join(MODELS_DIR, "emotion_ser_model.h5")
MODEL_PATH_GAN      = os.path.join(MODELS_DIR, "emotion_ser_model_gan.h5")
MODEL_PATH_FINETUNE = os.path.join(LEGACY_MODELS_DIR, "emotion_wavlm_finetuned.pt")
MODEL_PATH_FINETUNE_3DATASET = os.path.join(LEGACY_MODELS_DIR, "emotion_wavlm_finetuned_3dataset.pt")
MODEL_PATH_FINETUNE_3DATASET_V2 = os.path.join(MODELS_DIR, "emotion_wavlm_finetuned_3dataset_v2.pt")
MODEL_PATH_FINETUNE_DASHBOARD = (
    MODEL_PATH_FINETUNE_3DATASET_V2
    if os.path.exists(MODEL_PATH_FINETUNE_3DATASET_V2)
    else (MODEL_PATH_FINETUNE_3DATASET if os.path.exists(MODEL_PATH_FINETUNE_3DATASET) else MODEL_PATH_FINETUNE)
)
SCALER_PATH         = os.path.join(MODELS_DIR, "scaler.pkl")
ENCODER_PATH        = os.path.join(MODELS_DIR, "label_encoder.pkl")
GAN_GENERATOR_BEST  = os.path.join(MODELS_DIR, "gan_generator_best.h5")
GAN_DISCRIMINATOR   = os.path.join(MODELS_DIR, "gan_discriminator.h5")

# ── Model selection map (for dashboard) ──────────────────────────────────────
MODEL_OPTIONS = {
    "Original Model":     MODEL_PATH_BASE,
    "Fine-Tuned WavLM": MODEL_PATH_FINETUNE_DASHBOARD,
}

# ── Feature / architecture constants ────────────────────────────────────────
FEATURE_DIM  = 225
LATENT_DIM   = 100
NUM_EMOTIONS = 7

EMOTIONS = ["happy", "sad", "angry", "fear", "neutral", "surprise", "disgust"]
EMOTION_EMOJI = {
    "happy": "😊", "sad": "😢", "angry": "😠",
    "fear": "😨", "neutral": "🧘", "surprise": "😲", "disgust": "🤮",
}

# ── Feature names for feature importance panel ──────────────────────────────
FEATURE_GROUPS = {
    "MFCC Mean":          (0,   40),
    "MFCC Std":           (40,  80),
    "Delta MFCC":         (80, 120),
    "Delta² MFCC":       (120, 160),
    "Chroma":             (160, 172),
    "Spectral Contrast":  (172, 179),
    "Mel Spectrogram":    (179, 211),
    "ZCR":                (211, 212),
    "RMS Energy":         (212, 213),
    "Pitch Mean":         (213, 214),
    "Pitch Std":          (214, 215),
    "Pitch Range":        (215, 216),
    "Spectral Rolloff":   (216, 217),
    "Spectral Centroid":  (217, 218),
    "Tonnetz":            (218, 224),
    "Spectral Flatness":  (224, 225),
}

# ── Training defaults ───────────────────────────────────────────────────────
DEFAULT_EPOCHS     = 300
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR         = 0.0005
DEFAULT_DROPOUT    = 0.3

# ── Colors (dashboard theme) ────────────────────────────────────────────────
ACCENT       = "#63b3ed"
ACCENT2      = "#f6ad55"
ACCENT_PINK  = "#fc8181"
ACCENT_GREEN = "#68d391"
ACCENT_PURPLE = "#b794f4"

EMOTION_COLORS = {
    "happy":    "#f6ad55",
    "sad":      "#63b3ed",
    "angry":    "#fc8181",
    "fear":     "#b794f4",
    "neutral":  "#68d391",
    "surprise": "#fbd38d",
    "disgust":  "#81e6d9",
}
