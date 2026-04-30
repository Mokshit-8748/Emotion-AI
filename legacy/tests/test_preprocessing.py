import librosa.display
import matplotlib.pyplot as plt
from audio_preprocessing import load_and_preprocess_audio

file_path = "data/RAVDESS/Actor_01/03-01-05-01-01-01-01.wav"

y, sr = load_and_preprocess_audio(file_path)

plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr)
plt.title("Preprocessed Audio Waveform")
plt.show()
