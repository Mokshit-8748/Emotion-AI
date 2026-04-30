import numpy as np
import librosa
import torch
from torch.utils.data import Dataset


MAX_SAMPLES = 80000
EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
label_to_idx = {e: i for i, e in enumerate(EMOTION_CLASSES)}


class EmotionDataset(Dataset):
    def __init__(self, file_pairs, extractor, is_train=True):
        self.file_pairs = file_pairs
        self.extractor = extractor
        self.is_train = is_train
        self._failed = 0

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        fpath, emotion = self.file_pairs[idx]
        try:
            audio, sr = librosa.load(fpath, sr=16000, mono=True)

            if len(audio) < MAX_SAMPLES:
                audio = np.pad(audio, (0, MAX_SAMPLES - len(audio)))

            if len(audio) > MAX_SAMPLES:
                if self.is_train:
                    start = np.random.randint(0, len(audio) - MAX_SAMPLES)
                else:
                    start = (len(audio) - MAX_SAMPLES) // 2
                audio = audio[start:start + MAX_SAMPLES]

            if self.is_train:
                if np.random.rand() < 0.5:
                    noise = np.random.randn(len(audio)).astype(np.float32)
                    audio = audio + 0.002 * noise

                if np.random.rand() < 0.5:
                    gain = np.random.uniform(0.8, 1.2)
                    audio = audio * gain

                if np.random.rand() < 0.3:
                    rate = np.random.uniform(0.9, 1.1)
                    audio = librosa.effects.time_stretch(audio, rate=rate)
                    if len(audio) > MAX_SAMPLES:
                        start = np.random.randint(0, len(audio) - MAX_SAMPLES)
                        audio = audio[start:start + MAX_SAMPLES]
                    elif len(audio) < MAX_SAMPLES:
                        audio = np.pad(audio, (0, MAX_SAMPLES - len(audio)))

            audio = audio.astype(np.float32)
            label_idx = label_to_idx[emotion]
            return {"audio": audio, "label": label_idx}
        except Exception:
            self._failed += 1
            return None


class DataCollatorWithExtractor:
    def __init__(self, extractor):
        self.extractor = extractor

    def __call__(self, batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            return (
                torch.zeros(1, MAX_SAMPLES),
                torch.zeros(1, MAX_SAMPLES, dtype=torch.long),
                torch.zeros(1, dtype=torch.long),
            )
        audios = [b["audio"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        inputs = self.extractor(
            audios,
            sampling_rate=16000,
            return_tensors="pt",
            max_length=MAX_SAMPLES,
            padding="max_length",
            truncation=True,
        )
        return inputs.input_values, inputs.attention_mask, labels


def mixup_data(x, y, alpha=0.3):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
