import os

import numpy as np

from src.dataset_index import (
    CREMAD_MAP,
    EMODB_MAP,
    EMOTIONS,
    RAVDESS_MAP,
    SAVEE_MAP,
    TESS_MAP,
    collect_file_pairs,
    detect_dataset as _detect_dataset,
)
from src.feature_extraction import extract_features


AUGMENT_EMOTIONS = {"neutral", "happy", "sad", "angry", "fear", "surprise", "disgust"}
N_AUGMENTS = 0


def _append_features(X, y, file_path, emotion):
    if emotion not in EMOTIONS:
        return
    try:
        features = extract_features(file_path, augment=False)
        X.append(features)
        y.append(emotion)
        if emotion in AUGMENT_EMOTIONS:
            for _ in range(N_AUGMENTS):
                try:
                    features_aug = extract_features(file_path, augment=True)
                    X.append(features_aug)
                    y.append(emotion)
                except Exception:
                    pass
    except Exception:
        pass


def _load_single(dataset_path):
    X, y = [], []
    file_pairs, kind = collect_file_pairs(dataset_path, return_kind=True)
    print(f"  Detected: {kind}  ({dataset_path})")

    for file_path, emotion in file_pairs:
        _append_features(X, y, file_path, emotion)

    return np.array(X), np.array(y)


def build_dataset(dataset_path):
    """Build dataset from a single folder with cache support."""
    cache_path = os.path.join(dataset_path, "cached_features.npz")
    if os.path.exists(cache_path):
        print(f"  Loading cached features from: {cache_path}")
        data = np.load(cache_path)
        return data["X"], data["y"]

    X, y = _load_single(dataset_path)

    if len(X) > 0:
        print(f"  Saving features to cache: {cache_path}")
        np.savez_compressed(cache_path, X=X, y=y)

    return X, y


def build_multi_dataset(dataset_paths):
    """Build a combined feature dataset from multiple dataset roots."""
    all_X, all_y = [], []
    for path in dataset_paths:
        if not os.path.isdir(path):
            print(f"  Skipping {path} - folder not found.")
            continue
        X, y = build_dataset(path)
        print(f"  Loaded {len(X)} samples from {path}")
        all_X.append(X)
        all_y.extend(y)
    if not all_X:
        raise ValueError("No valid dataset folders found.")
    return np.vstack(all_X), np.array(all_y)
