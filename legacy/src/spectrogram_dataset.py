"""
Build a spectrogram dataset by walking RAVDESS / CREMA-D / TESS folders,
extracting mel spectrograms, and caching the result as .npz.
"""
import os
import numpy as np
from src.spectrogram_features import extract_mel_spectrogram
from src.dataset_builder import (
    EMOTIONS, RAVDESS_MAP, CREMAD_MAP, TESS_MAP, _detect_dataset,
)


def _collect_file_list(dataset_path):
    """Walk one dataset folder and return [(file_path, emotion), ...]."""
    from src.dataset_builder import EMODB_MAP, SAVEE_MAP
    
    # Optional IEMOCAP MAP to match our 7 existing classes
    IEMOCAP_MAP = {
        'ang': 'angry',
        'hap': 'happy', 
        'exc': 'happy',    # excited -> happy
        'sad': 'sad',
        'neu': 'neutral',
        'fru': 'angry',    # frustration -> angry
        'sur': 'surprise',
        'fea': 'fear',
        'dis': 'disgust',
    }

    kind = _detect_dataset(dataset_path)
    if kind == "unknown":
        print(f"  WARNING: could not detect dataset type for {dataset_path}")
        return [], kind

    pairs = []
    
    # ── EMO-DB is flat ───────────────────────────────────────────────────────
    if kind == "emodb":
        for fname in os.listdir(dataset_path):
            if fname.lower().endswith(".wav") and len(fname) >= 6 and fname[5] in EMODB_MAP:
                emotion = EMODB_MAP[fname[5]]
                if emotion in EMOTIONS:
                    pairs.append((os.path.join(dataset_path, fname), emotion))
        return pairs, kind
        
    # ── IEMOCAP is deeply nested ─────────────────────────────────────────────
    if kind == "iemocap":
        # Load IEMOCAP labels from Session folders
        utterance_map = {}
        for sess in range(1, 6):
            eval_dir = os.path.join(dataset_path, f"Session{sess}", "dialog", "EmoEvaluation")
            if os.path.exists(eval_dir):
                import re
                for txt_file in os.listdir(eval_dir):
                    if not txt_file.endswith(".txt"): continue
                    with open(os.path.join(eval_dir, txt_file), 'r', encoding='utf-8', errors='ignore') as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith('%') or line.startswith('C-'): continue
                            match = re.match(r'\[[\d\.]+ - [\d\.]+\]\s+(\S+)\s+(\S+)\s+\[', line)
                            if match:
                                utterance_map[match.group(1)] = match.group(2)
                                
        for root, dirs, files in os.walk(dataset_path):
            for fname in files:
                if fname.lower().endswith(".wav"):
                    utt_id = fname[:-4]
                    if utt_id in utterance_map:
                        raw_emotion = utterance_map[utt_id]
                        if raw_emotion in IEMOCAP_MAP:
                            mapped_emo = IEMOCAP_MAP[raw_emotion]
                            if mapped_emo in EMOTIONS:
                                pairs.append((os.path.join(root, fname), mapped_emo))
        return pairs, kind

    # ── Others have at least one layer of folders ────────────────────────────
    for folder in sorted(os.listdir(dataset_path)):
        folder_path = os.path.join(dataset_path, folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in sorted(os.listdir(folder_path)):
            if not fname.lower().endswith(".wav"):
                continue
            emotion = None
            if kind == "ravdess":
                parts = fname.split("-")
                if len(parts) >= 7 and parts[2] in RAVDESS_MAP:
                    emotion = RAVDESS_MAP[parts[2]]
            elif kind == "cremad":
                parts = fname.split("_")
                if len(parts) >= 3 and parts[2].upper() in CREMAD_MAP:
                    emotion = CREMAD_MAP[parts[2].upper()]
            elif kind == "tess":
                fl = folder.lower()
                for kw, em in TESS_MAP.items():
                    if kw in fl:
                        emotion = em
                        break
            elif kind == "savee":
                # fname prefix before digits
                import re
                match = re.match(r'([a-zA-Z]+)\d+', fname)
                if match:
                    prefix = match.group(1).lower()
                    if prefix in SAVEE_MAP:
                        emotion = SAVEE_MAP[prefix]
                        
            if emotion and emotion in EMOTIONS:
                pairs.append((os.path.join(folder_path, fname), emotion))
    return pairs, kind


def build_spectrogram_dataset(dataset_paths, cache_path="data/cached_spectrograms.npz"):
    """
    Build or load the full spectrogram dataset.

    Returns:
        X  np.ndarray  (N, 128, 128, 3)  float32 in [0,1]
        y  np.ndarray  (N,)              string labels
    """
    if os.path.exists(cache_path):
        print(f"  Loading cached spectrograms from: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data["X"], data["y"]

    # Collect all (file_path, emotion) pairs across datasets
    all_pairs = []
    for dpath in dataset_paths:
        if not os.path.isdir(dpath):
            print(f"  Skipping {dpath} — folder not found")
            continue
        pairs, kind = _collect_file_list(dpath)
        print(f"  {dpath}  ({kind}) → {len(pairs)} files")
        all_pairs.extend(pairs)

    print(f"\n  Total audio files: {len(all_pairs)}")
    print("  Extracting mel spectrograms (this takes a while on first run)...\n")

    all_X, all_y = [], []
    for i, (fpath, emotion) in enumerate(all_pairs, 1):
        try:
            spec = extract_mel_spectrogram(fpath)
            all_X.append(spec)
            all_y.append(emotion)
        except Exception as e:
            print(f"    [{i}] ERROR {os.path.basename(fpath)}: {e}")
        if i % 500 == 0 or i == len(all_pairs):
            print(f"    {i}/{len(all_pairs)}  ({i*100//len(all_pairs)}%)")

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y)

    print(f"\n  Saving cache → {cache_path}  (shape {X.shape})")
    np.savez_compressed(cache_path, X=X, y=y)

    return X, y
