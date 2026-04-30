import os
import re

# ── Extraction Patterns ───────────────────────────────────────────────────────
# Sync'd with src/dataset_builder.py logic for 100% accuracy.
RAVDESS_MAP = {"01":"neutral", "03":"happy", "04":"sad", "05":"angry", "06":"fear", "07":"disgust", "08":"surprise"}
CREMAD_MAP  = {"NEU":"neutral", "HAP":"happy", "SAD":"sad", "ANG":"angry", "FEA":"fear", "DIS":"disgust"}
TESS_MAP    = ["neutral", "happy", "sad", "angry", "fear", "disgust", "ps"] # ps = surprise

def get_fast_counts(dataset_paths):
    """
    Lightning-fast directory crawler for dataset headcounts.
    Bypasses audio loading/caching to provide instant metrics (<100ms).
    """
    counts = {emo: 0 for emo in ["happy", "sad", "angry", "fear", "neutral", "surprise", "disgust"]}
    
    for path in dataset_paths:
        if not os.path.exists(path): continue
        
        path_lower = path.lower()
        # Fast-detect dataset type by path or sample check
        kind = "ravdess" if "ravdess" in path_lower else ("crema" if "crema" in path_lower else ("tess" if "tess" in path_lower else None))
        
        for root, _, files in os.walk(path):
            # For TESS, the emotion is in the subfolder name
            folder_name = os.path.basename(root).lower()
            
            for f in files:
                if not f.endswith(".wav"): continue
                
                # ── RAVDESS ──
                if kind == "ravdess":
                    parts = f.split("-")
                    if len(parts) >= 3:
                        emo = RAVDESS_MAP.get(parts[2])
                        if emo: counts[emo] += 1
                
                # ── CREMA-D ──
                elif kind == "crema":
                    parts = f.split("_")
                    if len(parts) >= 3:
                        emo = CREMAD_MAP.get(parts[2].upper())
                        if emo: counts[emo] += 1
                
                # ── TESS ──
                elif kind == "tess":
                    for keyword in TESS_MAP:
                        if keyword in folder_name:
                            emo = "surprise" if keyword == "ps" else keyword
                            counts[emo] += 1
                            break
                            
    return counts
