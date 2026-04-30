import numpy as np
import librosa


def load_and_preprocess_audio(input_data, sr=None, augment=False):
    if isinstance(input_data, str):
        audio, sr = librosa.load(input_data, sr=None)
    else:
        audio = input_data.copy()

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    audio = audio.astype(np.float32)

    if len(audio) == 0:
        raise ValueError("Audio array is empty.")

    max_val = np.max(np.abs(audio))
    if max_val < 1e-9:
        raise ValueError("Audio is silent (all zeros).")

    audio = audio / (max_val + 1e-9)

    try:
        trimmed, _ = librosa.effects.trim(audio, top_db=20)
        if len(trimmed) > 0:
            audio = trimmed
    except Exception:
        pass

    if len(audio) == 0:
        raise ValueError("Audio is empty after silence trimming.")

    if augment:
        audio = _augment(audio, sr)

    return audio, sr


def _augment(audio, sr):
    """
    Randomly apply 1-2 augmentations per call.
    Simulates real-world acoustic variations to improve generalisation.
    """
    techniques = [_add_noise, _pitch_shift, _time_stretch,
                  _volume_change, _add_reverb]
    n_apply = np.random.randint(1, 3)
    chosen  = np.random.choice(len(techniques), size=n_apply, replace=False)

    for idx in chosen:
        try:
            audio = techniques[idx](audio, sr)
        except Exception:
            pass

    max_val = np.max(np.abs(audio))
    if max_val > 1e-9:
        audio = audio / max_val

    return audio


def _add_noise(audio, sr):
    """Gaussian noise — simulates mic/environment noise."""
    noise_level = np.random.uniform(0.002, 0.008)
    noise = np.random.randn(len(audio)).astype(np.float32)
    return audio + noise_level * noise


def _pitch_shift(audio, sr):
    """Shift pitch ±2.5 semitones — teaches pitch-invariant features."""
    semitones = np.random.uniform(-2.5, 2.5)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)


def _time_stretch(audio, sr):
    """Speed up/slow down ±20% — simulates fast/slow emotional speech."""
    rate = np.random.uniform(0.8, 1.2)
    stretched = librosa.effects.time_stretch(audio, rate=rate)
    if len(stretched) > len(audio):
        stretched = stretched[:len(audio)]
    elif len(stretched) < len(audio):
        stretched = np.pad(stretched, (0, len(audio) - len(stretched)))
    return stretched


def _volume_change(audio, sr):
    """Random volume ±30% — simulates mic distance variation."""
    factor = np.random.uniform(0.7, 1.3)
    return audio * factor


def _add_reverb(audio, sr):
    """Simple room reverb via decaying impulse response convolution."""
    decay   = np.random.uniform(0.3, 0.7)
    ir_len  = int(sr * np.random.uniform(0.05, 0.15))
    impulse = np.exp(-decay * np.arange(ir_len) / sr).astype(np.float32)
    impulse /= impulse.sum()
    reverbed = np.convolve(audio, impulse, mode='full')[:len(audio)]
    mix = np.random.uniform(0.3, 0.6)
    return (1 - mix) * audio + mix * reverbed.astype(np.float32)


def preprocess_audio(audio, sr):
    """Streamlit helper — no augmentation."""
    audio, _ = load_and_preprocess_audio(audio, sr, augment=False)
    return audio
