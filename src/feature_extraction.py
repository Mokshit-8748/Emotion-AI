import math
import numpy as np
import librosa

from src.audio_preprocessing import load_and_preprocess_audio


def _safe_stat_mean(feature):
    return np.mean(feature, axis=1 if feature.ndim > 1 else 0)


def _safe_stat_std(feature):
    return np.std(feature, axis=1 if feature.ndim > 1 else 0)


def _resolve_fft(audio_len):
    if audio_len < 256:
        return 256
    n_fft = min(2048, audio_len)
    power = int(math.floor(math.log2(max(256, n_fft))))
    return max(256, 2 ** power)


def _pitch_summary(audio, sr):
    try:
        fmax = min(600, max(120, sr // 2 - 1))
        f0 = librosa.yin(audio, fmin=50, fmax=fmax, sr=sr)
        voiced = f0[np.isfinite(f0)]
        voiced = voiced[voiced > 0]
        if voiced.size == 0:
            return np.zeros(3, dtype=np.float32)
        return np.array(
            [np.mean(voiced), np.std(voiced), np.max(voiced) - np.min(voiced)],
            dtype=np.float32,
        )
    except Exception:
        return np.zeros(3, dtype=np.float32)


def extract_features(input_data, sr=None, augment=False):
    audio, sr = load_and_preprocess_audio(input_data, sr=sr, augment=augment)

    # Some IEMOCAP utterances become extremely short after trimming.
    # Pad them to a safe minimum so downstream spectral ops stay stable.
    min_len = 2048
    if len(audio) < min_len:
        audio = np.pad(audio, (0, min_len - len(audio)))

    n_fft = _resolve_fft(len(audio))
    hop_length = max(128, n_fft // 4)

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40, n_fft=n_fft, hop_length=hop_length)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=32,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=hop_length)
    rms = librosa.feature.rms(y=audio, frame_length=n_fft, hop_length=hop_length)
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length)
    harmonic = librosa.effects.harmonic(audio)
    harmonic_chroma = librosa.feature.chroma_stft(
        y=harmonic, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    tonnetz = librosa.feature.tonnetz(chroma=harmonic_chroma, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=audio, n_fft=n_fft, hop_length=hop_length)
    pitch = _pitch_summary(audio, sr)

    feature_vector = np.concatenate(
        [
            _safe_stat_mean(mfcc),
            _safe_stat_std(mfcc),
            _safe_stat_mean(delta),
            _safe_stat_mean(delta2),
            _safe_stat_mean(chroma),
            _safe_stat_mean(contrast),
            _safe_stat_mean(mel_db),
            np.array([np.mean(zcr)], dtype=np.float32),
            np.array([np.mean(rms)], dtype=np.float32),
            pitch,
            np.array([np.mean(rolloff)], dtype=np.float32),
            np.array([np.mean(centroid)], dtype=np.float32),
            _safe_stat_mean(tonnetz),
            np.array([np.mean(flatness)], dtype=np.float32),
        ]
    ).astype(np.float32)

    if feature_vector.shape[0] != 225:
        raise ValueError(f"Expected 225 features, got {feature_vector.shape[0]}")

    return feature_vector
