import json
import os

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2FeatureExtractor

from src.audio_preprocessing import preprocess_audio
from src.wavlm_model import EmotionWavLM

MAX_SAMPLES = 80000
DEFAULT_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def load_feature_extractor(model_name="microsoft/wavlm-base-plus"):
    try:
        return Wav2Vec2FeatureExtractor.from_pretrained(model_name, local_files_only=True)
    except Exception:
        return Wav2Vec2FeatureExtractor.from_pretrained(model_name)


def _metadata_path_for_checkpoint(checkpoint_path):
    root, _ = os.path.splitext(checkpoint_path)
    candidate = root + ".json"
    return candidate if os.path.exists(candidate) else None


class WavLMRuntime:
    ser_backend = "wavlm_pt"

    def __init__(self, checkpoint_path, model, extractor, device, class_names=None, metadata=None):
        self.checkpoint_path = checkpoint_path
        self.model = model
        self.extractor = extractor
        self.device = device
        self.class_names = class_names or list(DEFAULT_CLASSES)
        self.metadata = metadata or {}
        self._warmed_up = False

    def prepare_audio(self, audio, sr):
        audio = np.asarray(audio)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = audio.astype(np.float32)
        audio = preprocess_audio(audio, sr)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000

        if len(audio) < MAX_SAMPLES:
            audio = np.pad(audio, (0, MAX_SAMPLES - len(audio)))
        elif len(audio) > MAX_SAMPLES:
            start = (len(audio) - MAX_SAMPLES) // 2
            audio = audio[start:start + MAX_SAMPLES]

        return audio.astype(np.float32), sr

    def _predict_batch(self, audio_batch):
        inputs = self.extractor(
            audio_batch,
            sampling_rate=16000,
            return_tensors="pt",
            max_length=MAX_SAMPLES,
            padding="max_length",
            truncation=True,
        )
        input_values = inputs.input_values.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        with torch.inference_mode():
            logits = self.model(input_values, attention_mask)
            probs = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    def warmup(self):
        """Run a tiny silent pass once so first real inference feels faster."""
        if self._warmed_up:
            return
        silence = np.zeros(MAX_SAMPLES, dtype=np.float32)
        self._predict_batch([silence])
        self._warmed_up = True

    def predict_audio(self, audio, sr):
        prepared, prepared_sr = self.prepare_audio(audio, sr)
        probs = self._predict_batch([prepared])[0]
        pred_idx = int(np.argmax(probs))
        return {
            "audio": prepared,
            "sr": prepared_sr,
            "emotion": self.class_names[pred_idx],
            "confidence": float(np.max(probs)) * 100.0,
            "probabilities": probs,
            "labels": list(self.class_names),
            "input_samples": int(len(prepared)),
        }

    def predict_file_pairs(self, file_pairs, batch_size=4):
        label_to_idx = {label: idx for idx, label in enumerate(self.class_names)}
        usable_pairs = [(path, label) for path, label in file_pairs if label in label_to_idx]

        y_true = []
        probabilities = []

        for start in range(0, len(usable_pairs), batch_size):
            batch_pairs = usable_pairs[start:start + batch_size]
            batch_audio = []
            batch_labels = []

            for file_path, label in batch_pairs:
                try:
                    audio, sr = librosa.load(file_path, sr=16000, mono=True)
                    prepared, _ = self.prepare_audio(audio, sr)
                    batch_audio.append(prepared)
                    batch_labels.append(label_to_idx[label])
                except Exception:
                    continue

            if not batch_audio:
                continue

            probs = self._predict_batch(batch_audio)
            probabilities.append(probs)
            y_true.extend(batch_labels)

        if not probabilities:
            return np.empty((0,), dtype=np.int64), np.empty((0, len(self.class_names)), dtype=np.float32)

        return np.asarray(y_true, dtype=np.int64), np.vstack(probabilities).astype(np.float32)


def load_wavlm_runtime(checkpoint_path, class_names=None, device=None):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metadata = {}
    metadata_path = _metadata_path_for_checkpoint(checkpoint_path)
    if metadata_path:
        try:
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except Exception:
            metadata = {}

    args = metadata.get("args", {})
    unfreeze_layers = int(args.get("unfreeze_layers", 12))
    extractor = load_feature_extractor("microsoft/wavlm-base-plus")
    model = EmotionWavLM(num_classes=len(class_names or DEFAULT_CLASSES), unfreeze_layers=unfreeze_layers)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    runtime = WavLMRuntime(
        checkpoint_path=checkpoint_path,
        model=model,
        extractor=extractor,
        device=device,
        class_names=list(class_names or DEFAULT_CLASSES),
        metadata=metadata,
    )
    try:
        runtime.warmup()
    except Exception:
        # Warmup is a UX optimization only; model loading should still succeed.
        pass
    return runtime
