"""Shared dataset detection and file indexing helpers."""

from __future__ import annotations

import os
import re
from typing import List, Tuple


EMOTIONS = {"neutral", "happy", "sad", "angry", "fear", "surprise", "disgust"}

RAVDESS_MAP = {
    "01": "neutral",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust",
    "08": "surprise",
}

CREMAD_MAP = {
    "NEU": "neutral",
    "HAP": "happy",
    "SAD": "sad",
    "ANG": "angry",
    "FEA": "fear",
    "DIS": "disgust",
}

TESS_MAP = {
    "neutral": "neutral",
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fear": "fear",
    "disgust": "disgust",
    "pleasant": "surprise",
}

EMODB_MAP = {
    "W": "angry",
    "L": "boredom",
    "E": "disgust",
    "A": "fear",
    "F": "happy",
    "T": "sad",
    "N": "neutral",
}

SAVEE_MAP = {
    "a": "angry",
    "d": "disgust",
    "f": "fear",
    "h": "happy",
    "n": "neutral",
    "sa": "sad",
    "su": "surprise",
}

IEMOCAP_MAP = {
    "ang": "angry",
    "hap": "happy",
    "exc": "happy",
    "sad": "sad",
    "neu": "neutral",
    "fru": "angry",
    "sur": "surprise",
    "fea": "fear",
    "dis": "disgust",
}


def detect_dataset(folder_path: str) -> str:
    """Infer dataset family from a dataset root folder."""
    if not os.path.isdir(folder_path):
        return "unknown"

    entries = os.listdir(folder_path)
    has_wav_root = any(name.lower().endswith(".wav") for name in entries)
    if has_wav_root:
        for name in entries:
            if name.lower().endswith(".wav") and len(name) >= 6 and name[5] in EMODB_MAP:
                return "emodb"

    if any(name.startswith("Session") for name in entries):
        return "iemocap"

    for entry in entries:
        entry_path = os.path.join(folder_path, entry)
        if not os.path.isdir(entry_path):
            continue
        if entry in ["DC", "JE", "JK", "KL"]:
            return "savee"

        for file_name in os.listdir(entry_path):
            if not file_name.lower().endswith(".wav"):
                continue
            parts_dash = file_name.split("-")
            parts_under = file_name.split("_")
            if len(parts_dash) >= 7 and parts_dash[0].isdigit():
                return "ravdess"
            if len(parts_under) >= 3 and parts_under[2].upper() in CREMAD_MAP:
                return "cremad"
            folder_lower = entry.lower()
            for keyword in TESS_MAP:
                if keyword in folder_lower:
                    return "tess"
    return "unknown"


def _collect_iemocap_pairs(dataset_path: str) -> List[Tuple[str, str]]:
    utterance_map = {}
    for sess in range(1, 6):
        eval_dir = os.path.join(dataset_path, f"Session{sess}", "dialog", "EmoEvaluation")
        if not os.path.isdir(eval_dir):
            continue
        for txt_file in os.listdir(eval_dir):
            if not txt_file.endswith(".txt"):
                continue
            with open(os.path.join(eval_dir, txt_file), "r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    line = line.strip()
                    if not line or line.startswith("%") or line.startswith("C-"):
                        continue
                    match = re.match(r"\[[\d\.]+ - [\d\.]+\]\s+(\S+)\s+(\S+)\s+\[", line)
                    if match:
                        utterance_map[match.group(1)] = match.group(2)

    pairs: List[Tuple[str, str]] = []
    for root, _, files in os.walk(dataset_path):
        for file_name in files:
            if not file_name.lower().endswith(".wav"):
                continue
            utterance_id = file_name[:-4]
            raw_emotion = utterance_map.get(utterance_id)
            emotion = IEMOCAP_MAP.get(raw_emotion)
            if emotion in EMOTIONS:
                pairs.append((os.path.join(root, file_name), emotion))
    return pairs


def collect_file_pairs(dataset_path: str, *, return_kind: bool = False):
    """Return a dataset's `(file_path, emotion)` pairs."""
    kind = detect_dataset(dataset_path)
    pairs: List[Tuple[str, str]] = []

    if kind == "unknown":
        return (pairs, kind) if return_kind else pairs

    if kind == "emodb":
        for file_name in os.listdir(dataset_path):
            if file_name.lower().endswith(".wav") and len(file_name) >= 6:
                emotion = EMODB_MAP.get(file_name[5])
                if emotion in EMOTIONS:
                    pairs.append((os.path.join(dataset_path, file_name), emotion))
        return (pairs, kind) if return_kind else pairs

    if kind == "iemocap":
        pairs = _collect_iemocap_pairs(dataset_path)
        return (pairs, kind) if return_kind else pairs

    for folder in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue
        for file_name in sorted(os.listdir(folder_path)):
            if not file_name.lower().endswith(".wav"):
                continue

            emotion = None
            if kind == "ravdess":
                parts = file_name.split("-")
                if len(parts) >= 7:
                    emotion = RAVDESS_MAP.get(parts[2])
            elif kind == "cremad":
                parts = file_name.split("_")
                if len(parts) >= 3:
                    emotion = CREMAD_MAP.get(parts[2].upper())
            elif kind == "tess":
                folder_lower = folder.lower()
                for keyword, mapped_emotion in TESS_MAP.items():
                    if keyword in folder_lower:
                        emotion = mapped_emotion
                        break
            elif kind == "savee":
                stem = os.path.splitext(file_name)[0].lower()
                match = re.match(r"(su|sa|[adfhn])\d+", stem)
                prefix = match.group(1) if match else None
                emotion = SAVEE_MAP.get(prefix)

            if emotion in EMOTIONS:
                pairs.append((os.path.join(folder_path, file_name), emotion))

    return (pairs, kind) if return_kind else pairs
