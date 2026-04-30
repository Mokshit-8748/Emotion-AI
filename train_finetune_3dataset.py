import argparse
from contextlib import nullcontext
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import Wav2Vec2FeatureExtractor

from config import DATASET_PATHS, MODEL_PATH_FINETUNE, PROJECT_ROOT
from src.dataset_index import collect_file_pairs
from src.wavlm_training_shared import (
    DataCollatorWithExtractor,
    EMOTION_CLASSES,
    EmotionDataset,
    get_scheduler,
    mixup_criterion,
    mixup_data,
)
from src.wavlm_model import EmotionWavLM


DEFAULT_DATASETS = DATASET_PATHS[:3]
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "models", "emotion_wavlm_finetuned_3dataset.pt")
DEFAULT_METADATA = os.path.join(PROJECT_ROOT, "models", "emotion_wavlm_finetuned_3dataset.json")
DEFAULT_INIT = MODEL_PATH_FINETUNE if os.path.exists(MODEL_PATH_FINETUNE) else ""


def load_feature_extractor(model_name="microsoft/wavlm-base-plus"):
    try:
        return Wav2Vec2FeatureExtractor.from_pretrained(model_name, local_files_only=True)
    except Exception:
        print(f"[cache-miss] Falling back to online load for {model_name}")
        return Wav2Vec2FeatureExtractor.from_pretrained(model_name)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="3-dataset WavLM fine-tuning mode.")
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated dataset paths. Defaults to RAVDESS, CREMA-D, TESS.",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--init-from", default=DEFAULT_INIT, help="Optional checkpoint to warm-start model weights.")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--lr-backbone", type=float, default=5e-6)
    parser.add_argument("--lr-head", type=float, default=2e-4)
    parser.add_argument("--lr-layer-weights", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--mixup-prob", type=float, default=0.45)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--unfreeze-layers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--balance-sampler", action="store_true", default=True)
    parser.add_argument("--no-balance-sampler", dest="balance_sampler", action="store_false")
    parser.add_argument("--quick-run", action="store_true")
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    return parser.parse_args()


def maybe_limit_pairs(pairs, limit, seed):
    if not limit or limit <= 0 or len(pairs) <= limit:
        return pairs
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(pairs), size=limit, replace=False)
    indices = np.sort(indices)
    return [pairs[i] for i in indices]


def collect_pairs(dataset_paths):
    all_pairs = []
    per_dataset = {}

    for dpath in dataset_paths:
        if not os.path.exists(dpath):
            print(f"[skip] Missing dataset path: {dpath}")
            continue
        pairs = collect_file_pairs(dpath)
        all_pairs.extend(pairs)
        per_dataset[dpath] = len(pairs)
        print(f"  {dpath} -> {len(pairs)} usable files")

    if not all_pairs:
        raise RuntimeError("No audio files found for the selected datasets.")

    return all_pairs, per_dataset


def make_sampler(train_pairs):
    label_to_idx = {e: i for i, e in enumerate(EMOTION_CLASSES)}
    labels = np.array([label_to_idx[emo] for _, emo in train_pairs], dtype=np.int64)
    class_counts = np.bincount(labels, minlength=len(EMOTION_CLASSES)).astype(np.float64)
    class_counts[class_counts == 0] = 1.0
    class_weights = class_counts.sum() / (len(class_counts) * class_counts)
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler, class_counts


def save_metadata(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main():
    args = parse_args()
    if args.quick_run:
        args.epochs = 1
        args.batch_size = 2
        args.grad_accum_steps = 2
        args.num_workers = 0
        args.limit_train = 64 if args.limit_train == 0 else args.limit_train
        args.limit_val = 32 if args.limit_val == 0 else args.limit_val

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    if args.device == "cpu":
        device = torch.device("cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no GPU is available.")
        device = torch.device("cuda")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_paths = [p.strip() for p in args.datasets.split(",") if p.strip()]

    print("=" * 72)
    print("  3-DATASET WAVLM FINE-TUNING MODE")
    print("=" * 72)
    print(f"Device: {device}")
    print(f"Datasets: {dataset_paths}")

    all_pairs, per_dataset = collect_pairs(dataset_paths)
    labels = np.array([emo for _, emo in all_pairs])

    train_pairs, val_pairs = train_test_split(
        all_pairs,
        test_size=args.val_split,
        random_state=args.seed,
        stratify=labels,
    )

    train_pairs = maybe_limit_pairs(train_pairs, args.limit_train, args.seed)
    val_pairs = maybe_limit_pairs(val_pairs, args.limit_val, args.seed)

    print(f"Train files: {len(train_pairs)} | Val files: {len(val_pairs)}")

    extractor = load_feature_extractor("microsoft/wavlm-base-plus")
    collator = DataCollatorWithExtractor(extractor)

    train_ds = EmotionDataset(train_pairs, extractor, is_train=True)
    val_ds = EmotionDataset(val_pairs, extractor, is_train=False)

    sampler = None
    class_counts = None
    if args.balance_sampler:
        sampler, class_counts = make_sampler(train_pairs)
        print(f"Balanced sampler class counts: {class_counts.astype(int).tolist()}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
        drop_last=False,
    )

    model = EmotionWavLM(num_classes=len(EMOTION_CLASSES), unfreeze_layers=args.unfreeze_layers)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.get_param_groups(args.lr_backbone, args.lr_head, args.lr_layer_weights),
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum_steps))
    total_steps = max(1, args.epochs * steps_per_epoch)
    warmup_steps = max(1, args.warmup_epochs * steps_per_epoch)
    scheduler = get_scheduler(optimizer, warmup_steps, total_steps)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

    best_val_acc = 0.0
    start_epoch = 0
    patience_counter = 0

    if os.path.exists(args.output):
        checkpoint = torch.load(args.output, map_location=device, weights_only=False)
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            best_val_acc = checkpoint.get("val_acc", 0.0)
            start_epoch = checkpoint.get("epoch", 0) + 1
            print(f"[resume] epoch={start_epoch} best_val_acc={best_val_acc:.2f}%")
        except Exception as exc:
            print(f"[resume] Could not restore checkpoint cleanly: {exc}")
    elif args.init_from and os.path.exists(args.init_from):
        init_checkpoint = torch.load(args.init_from, map_location=device, weights_only=False)
        init_state = init_checkpoint.get("model_state_dict", init_checkpoint)
        load_result = model.load_state_dict(init_state, strict=False)
        missing = len(getattr(load_result, "missing_keys", []))
        unexpected = len(getattr(load_result, "unexpected_keys", []))
        init_val_acc = init_checkpoint.get("val_acc")
        init_msg = f"[init] warm-started model weights from {args.init_from}"
        if init_val_acc is not None:
            init_msg += f" (source val_acc={init_val_acc:.2f}%)"
        init_msg += f" | missing_keys={missing} unexpected_keys={unexpected}"
        print(init_msg)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    autocast_ctx = (
        (lambda: torch.amp.autocast(device_type="cuda"))
        if device.type == "cuda"
        else nullcontext
    )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        correct = 0.0
        total = 0
        optimizer.zero_grad()
        grad_accum_count = 0

        for batch_idx, (inputs, attention_mask, labels_batch) in enumerate(train_loader):
            if inputs.shape[0] == 0:
                continue

            inputs = inputs.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True)
            labels_batch = labels_batch.to(device, non_blocking=True)

            if np.random.rand() < args.mixup_prob:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels_batch, args.mixup_alpha)
            else:
                targets_a, targets_b, lam = labels_batch, labels_batch, 1.0

            with autocast_ctx():
                outputs = model(inputs, attention_mask)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                loss = loss / args.grad_accum_steps

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            train_loss += loss.item() * args.grad_accum_steps
            _, predicted = outputs.max(1)
            total += labels_batch.size(0)
            if lam < 1.0:
                correct += (
                    lam * predicted.eq(targets_a).float()
                    + (1 - lam) * predicted.eq(targets_b).float()
                ).sum().item()
            else:
                correct += predicted.eq(targets_a).sum().item()

            grad_accum_count += 1
            if grad_accum_count % args.grad_accum_steps == 0:
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

            if batch_idx % 50 == 0:
                current_lr = optimizer.param_groups[2]["lr"]
                print(
                    f"epoch={epoch + 1:02d} batch={batch_idx:04d}/{len(train_loader)} "
                    f"loss={loss.item() * args.grad_accum_steps:.4f} lr_head={current_lr:.2e}",
                    flush=True,
                )

        if grad_accum_count % args.grad_accum_steps != 0:
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

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, attention_mask, labels_batch in val_loader:
                if inputs.shape[0] == 0:
                    continue
                inputs = inputs.to(device, non_blocking=True)
                attention_mask = attention_mask.to(device, non_blocking=True)
                labels_batch = labels_batch.to(device, non_blocking=True)

                with autocast_ctx():
                    outputs = model(inputs, attention_mask)
                    loss = criterion(outputs, labels_batch)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels_batch.size(0)
                val_correct += predicted.eq(labels_batch).sum().item()

        train_acc = 100.0 * correct / total if total > 0 else 0.0
        val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0.0
        avg_train_loss = train_loss / max(1, len(train_loader))
        avg_val_loss = val_loss / max(1, len(val_loader))

        print("-" * 72)
        print(
            f"EPOCH {epoch + 1:02d}: "
            f"train_loss={avg_train_loss:.4f} train_acc={train_acc:.2f}% "
            f"val_loss={avg_val_loss:.4f} val_acc={val_acc:.2f}%"
        )
        print("-" * 72)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_acc": val_acc,
                    "datasets": dataset_paths,
                    "per_dataset_counts": per_dataset,
                    "init_from": args.init_from,
                    "args": vars(args),
                },
                args.output,
            )
            save_metadata(
                args.metadata,
                {
                    "best_val_acc": best_val_acc,
                    "epoch": epoch + 1,
                    "datasets": dataset_paths,
                    "per_dataset_counts": per_dataset,
                    "init_from": args.init_from,
                    "args": vars(args),
                },
            )
            print(f"[best] saved checkpoint to {args.output}")
        else:
            patience_counter += 1
            print(f"[patience] {patience_counter}/{args.patience}")
            if patience_counter >= args.patience:
                print(f"[early-stop] Best validation accuracy: {best_val_acc:.2f}%")
                break

    print("=" * 72)
    print(f"Training complete. Best validation accuracy: {best_val_acc:.2f}%")
    print("=" * 72)


if __name__ == "__main__":
    main()
