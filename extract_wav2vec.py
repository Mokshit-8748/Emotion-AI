"""
extract_wav2vec.py (Streaming HDF5 v4 - 90% Accuracy Target)
Extracts a SEQUENCE of 768-D contextual embeddings using Microsoft WavLM Base+.
FIXED: Uses HDF5 to stream data directly to disk (solves 10.5GB RAM wall).
FIXED: 320-frame high-resolution sequence preservation.
"""
import os
import sys
import numpy as np
import librosa
import torch
import h5py
from transformers import Wav2Vec2FeatureExtractor, WavLMModel
from config import DATASET_PATHS
from src.dataset_index import collect_file_pairs

def extract_features(
    h5_path="data/cached_wavlm_seq.h5",
    batch_size=4,
    max_seq_len=320,
    dataset_paths=None,
    rebuild=False,
):
    if dataset_paths is None:
        dataset_paths = DATASET_PATHS

    if os.path.exists(h5_path):
        if rebuild:
            print(f"[!] Found old cache. Deleting to rebuild: {h5_path}")
            os.remove(h5_path)
        else:
            print(f"[OK] Using existing cache: {h5_path}")
            return h5_path
        
    print(f"Loading Pre-trained WavLM Model...")
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    all_pairs = []
    print(f"Scanning dataset paths...")
    for dpath in dataset_paths:
        if not os.path.exists(dpath): continue
        pairs, kind = collect_file_pairs(dpath, return_kind=True)
        all_pairs.extend(pairs)
        
    if not all_pairs:
        print("\n[CRITICAL] No audio files found!")
        return

    num_files = len(all_pairs)
    print(f"\nTotal files: {num_files}. Initializing HDF5 stream...")
    
    # Initialize HDF5 File
    os.makedirs(os.path.dirname(h5_path), exist_ok=True)
    with h5py.File(h5_path, 'w') as f:
        # Create datasets (Batch size doesn't matter for disk allocation)
        X_ds = f.create_dataset('X', (num_files, max_seq_len, 768), dtype='float32')
        # Labels need fixed-length bytes for speed
        y_ds = f.create_dataset('y', (num_files,), dtype='S20') 
        
        # Process batches
        for i in range(0, num_files, batch_size):
            batch_pairs = all_pairs[i:i+batch_size]
            batch_audio = []
            batch_y = []
            
            for fpath, emotion in batch_pairs:
                try:
                    audio, sr = librosa.load(fpath, sr=16000)
                    target_samples = 100000 
                    if len(audio) > target_samples:
                        audio = audio[:target_samples]
                    else:
                        audio = np.pad(audio, (0, target_samples - len(audio)))
                    batch_audio.append(audio)
                    batch_y.append(emotion.encode('utf-8'))
                except Exception:
                    # Pad empty for failed files to keep index alignment
                    batch_audio.append(np.zeros(100000))
                    batch_y.append(b'unknown')
            
            inputs = feature_extractor(batch_audio, sampling_rate=16000, return_tensors="pt", padding=True).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                hidden_states = outputs.last_hidden_state
                curr_len = hidden_states.shape[1]
                
                if curr_len > max_seq_len:
                    sampled_features = hidden_states[:, :max_seq_len, :]
                else:
                    padding_size = max_seq_len - curr_len
                    sampled_features = torch.nn.functional.pad(hidden_states, (0, 0, 0, padding_size))
            
            # WRITE DIRECTLY TO DISK (Zero RAM accumulation)
            X_ds[i:i+len(batch_pairs)] = sampled_features.cpu().numpy()
            y_ds[i:i+len(batch_pairs)] = batch_y
            
            if (i + batch_size) % 100 < batch_size:
                print(f"  Streaming: {min(i+batch_size, num_files)} / {num_files} to HDF5...")

    print(f"Finished! Dataset saved to {h5_path}")
    return h5_path

if __name__ == "__main__":
    extract_features(rebuild=True)
