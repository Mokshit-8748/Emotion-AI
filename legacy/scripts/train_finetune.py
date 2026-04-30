import os
import gc
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2FeatureExtractor
import warnings

warnings.filterwarnings('ignore')

from src.wavlm_model import EmotionWavLM
from src.dataset_index import collect_file_pairs
from config import DATASET_PATHS, MODEL_PATH_FINETUNE, PROJECT_ROOT


def _collect_file_list(dataset_path):
    return collect_file_pairs(dataset_path)


# ═══════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ═══════════════════════════════════════════════════════════════════════
MAX_SAMPLES        = 80000   # 5 seconds at 16kHz
BATCH_SIZE         = 16
GRAD_ACCUM_STEPS   = 4       # effective batch size = 64
EPOCHS             = 60
LR_BACKBONE        = 5e-6
LR_HEAD            = 3e-4
LR_LAYER_WEIGHTS   = 5e-6    # FIX: layer_weights gets backbone-scale LR, not head-scale
WEIGHT_DECAY       = 0.01
LABEL_SMOOTHING    = 0.05
MIXUP_ALPHA        = 0.3
WARMUP_EPOCHS      = 3
UNFREEZE_LAYERS    = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
num_classes     = len(EMOTION_CLASSES)
label_to_idx    = {e: i for i, e in enumerate(EMOTION_CLASSES)}


class EmotionDataset(Dataset):
    def __init__(self, file_pairs, extractor, is_train=True):
        self.file_pairs = file_pairs
        self.extractor  = extractor
        self.is_train   = is_train
        # FIX: track and report failed files rather than silently mislabelling them
        self._failed    = 0

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
                # 1. Gaussian noise
                if np.random.rand() < 0.5:
                    noise = np.random.randn(len(audio)).astype(np.float32)
                    audio = audio + 0.002 * noise

                # 2. Random gain
                if np.random.rand() < 0.5:
                    gain = np.random.uniform(0.8, 1.2)
                    audio = audio * gain

                # 3. Time stretch
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

        except Exception as e:
            # FIX: return None — collator will filter these out.
            # Do NOT return a fake neutral label — it poisons training.
            self._failed += 1
            return None


class DataCollatorWithExtractor:
    def __init__(self, extractor):
        self.extractor = extractor

    def __call__(self, batch):
        # FIX: filter out None items from failed loads
        batch = [b for b in batch if b is not None]
        if not batch:
            # return empty tensors — training loop will skip gracefully
            return torch.zeros(1, MAX_SAMPLES), torch.zeros(1, MAX_SAMPLES, dtype=torch.long), torch.zeros(1, dtype=torch.long)
        audios = [b["audio"] for b in batch]
        labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        inputs = self.extractor(audios, sampling_rate=16000, return_tensors="pt",
                                max_length=MAX_SAMPLES, padding="max_length", truncation=True)
        return inputs.input_values, inputs.attention_mask, labels


def mixup_data(x, y, alpha=0.3):
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ═══════════════════════════════════════════════════════════════════════
# SCHEDULER: Linear Warmup → Cosine Decay
# ═══════════════════════════════════════════════════════════════════════
def get_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train():
    print("=" * 60)
    print("  PHASE 2: END-TO-END WAVLM FINE-TUNING (v2.1)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Effective batch size: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    print(f"Unfrozen WavLM layers: {UNFREEZE_LAYERS}")

    # ── Load dataset ──────────────────────────────────────────────────
    print("\nCollecting dataset files...")
    all_pairs = []
    for dpath in DATASET_PATHS:
        if not os.path.exists(dpath): continue
        pairs = _collect_file_list(dpath)
        all_pairs.extend(pairs)
        print(f"  {dpath} -> {len(pairs)} usable files")

    if not all_pairs:
        print("[CRITICAL] No audio files found!")
        return

    np.random.seed(42)
    np.random.shuffle(all_pairs)
    split_idx   = int(len(all_pairs) * 0.8)
    train_pairs = all_pairs[:split_idx]
    val_pairs   = all_pairs[split_idx:]
    print(f"\nTotal: {len(all_pairs)} (Train: {len(train_pairs)}, Val: {len(val_pairs)})")

    # ── Loaders ───────────────────────────────────────────────────────
    extractor    = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    collator     = DataCollatorWithExtractor(extractor)

    train_ds     = EmotionDataset(train_pairs, extractor, is_train=True)
    val_ds       = EmotionDataset(val_pairs,   extractor, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2,
                              persistent_workers=True, collate_fn=collator, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
                              persistent_workers=True, collate_fn=collator)

    # ── Model ─────────────────────────────────────────────────────────
    print("\nInitializing EmotionWavLM...")
    model = EmotionWavLM(num_classes=num_classes, unfreeze_layers=UNFREEZE_LAYERS)
    model.to(device)

    # ── Optimizer with 3 separate LR groups ───────────────────────────
    # FIX: use model.get_param_groups() so layer_weights gets its own
    # backbone-scale LR (5e-6) instead of head-scale LR (3e-4).
    optimizer = torch.optim.AdamW(
        model.get_param_groups(LR_BACKBONE, LR_HEAD, LR_LAYER_WEIGHTS),
        weight_decay=WEIGHT_DECAY
    )

    # FIX: compute steps correctly — steps_per_epoch = optimizer updates per epoch
    steps_per_epoch = len(train_loader) // GRAD_ACCUM_STEPS
    total_steps     = EPOCHS * steps_per_epoch
    warmup_steps    = WARMUP_EPOCHS * steps_per_epoch
    scheduler       = get_scheduler(optimizer, warmup_steps, total_steps)

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scaler    = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

    # ── Checkpoint Resume ─────────────────────────────────────────────
    # FIX: restore optimizer AND scheduler state, not just model weights.
    # Without this, AdamW's moment estimates reset and the LR schedule
    # restarts from 0, causing the same ceiling problem as before.
    best_val_acc     = 0.0
    start_epoch      = 0
    patience_counter = 0

    if os.path.exists(MODEL_PATH_FINETUNE):
        checkpoint = torch.load(MODEL_PATH_FINETUNE, map_location=device)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print("[✓] Optimizer state restored.")
                except Exception as oe:
                    print(f"[!] Optimizer state mismatch (param groups changed?): {oe}")
                    print("[!] Proceeding with fresh optimizer — expect 2-3 epochs re-stabilization.")
            if 'scheduler_state_dict' in checkpoint:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    print("[✓] Scheduler state restored.")
                except Exception as se:
                    print(f"[!] Scheduler state mismatch: {se}")
            best_val_acc = checkpoint.get('val_acc', 0.0)
            start_epoch  = checkpoint.get('epoch', 0) + 1
            print(f"[✓] Resumed from epoch {start_epoch} (best val_acc={best_val_acc:.2f}%)")
        except Exception as e:
            print(f"[!] Could not resume: {e}. Starting fresh.")

    os.makedirs(os.path.dirname(MODEL_PATH_FINETUNE), exist_ok=True)
    PATIENCE = 12

    print("\nStarting Training...")
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss = 0.0
        correct    = 0.0
        total      = 0
        optimizer.zero_grad()
        grad_accum_count = 0  # track accumulated steps within epoch

        for batch_idx, (inputs, attention_mask, labels) in enumerate(train_loader):
            if inputs.shape[0] == 0:
                continue  # skip empty batches from failed file loads

            inputs, attention_mask, labels = (
                inputs.to(device), attention_mask.to(device), labels.to(device)
            )

            if np.random.rand() < 0.6:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, MIXUP_ALPHA)
            else:
                targets_a, targets_b, lam = labels, labels, 1.0

            # FIX: cleaner autocast — use device.type, works for both cuda and cpu
            with torch.amp.autocast(device_type=device.type):
                outputs = model(inputs, attention_mask)
                loss    = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                loss    = loss / GRAD_ACCUM_STEPS

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            train_loss += loss.item() * GRAD_ACCUM_STEPS
            _, predicted = outputs.max(1)
            total += labels.size(0)
            if lam < 1.0:
                correct += (lam * predicted.eq(targets_a).float() + (1 - lam) * predicted.eq(targets_b).float()).sum().item()
            else:
                correct += predicted.eq(targets_a).sum().item()

            grad_accum_count += 1

            if grad_accum_count % GRAD_ACCUM_STEPS == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            if batch_idx % 20 == 0:
                current_lr_head = optimizer.param_groups[2]['lr']  # head is now group index 2
                print(f"  Epoch: {epoch+1:02d} | Batch: {batch_idx:03d}/{len(train_loader)} "
                      f"| Loss: {loss.item() * GRAD_ACCUM_STEPS:.4f} | LR(head): {current_lr_head:.2e}")

        # FIX: flush any leftover accumulated gradients at end of epoch
        if grad_accum_count % GRAD_ACCUM_STEPS != 0:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # ── Validation ────────────────────────────────────────────────
        model.eval()
        val_loss    = 0.0
        val_correct = 0
        val_total   = 0

        with torch.no_grad():
            for inputs, attention_mask, labels in val_loader:
                if inputs.shape[0] == 0:
                    continue
                inputs, attention_mask, labels = (
                    inputs.to(device), attention_mask.to(device), labels.to(device)
                )
                with torch.amp.autocast(device_type=device.type):
                    outputs = model(inputs, attention_mask)
                    loss    = criterion(outputs, labels)
                val_loss    += loss.item()
                _, predicted = outputs.max(1)
                val_total   += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        train_acc      = 100. * correct / total if total > 0 else 0.0
        val_acc        = 100. * val_correct / val_total if val_total > 0 else 0.0
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss   = val_loss   / len(val_loader)

        print("-" * 60)
        print(f"EPOCH {epoch+1:02d} SUMMARY:")
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {avg_val_loss:.4f} | Val   Acc: {val_acc:.2f}%")
        if train_ds._failed > 0:
            print(f"[!] Failed audio loads this run: {train_ds._failed} (silently skipped)")
        print("-" * 60)

        if val_acc > best_val_acc:
            best_val_acc     = val_acc
            patience_counter = 0
            print(f"==> New Best! Saving to {MODEL_PATH_FINETUNE}")
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),   # FIX: save scheduler too
                'val_acc':              val_acc,
            }, MODEL_PATH_FINETUNE)
        else:
            patience_counter += 1
            print(f"    No improvement. Patience: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print(f"\n[Early Stop] No improvement for {PATIENCE} epochs. Best: {best_val_acc:.2f}%")
                break

    print(f"\nTraining complete. Best Val Acc: {best_val_acc:.2f}%")


if __name__ == "__main__":
    train()
