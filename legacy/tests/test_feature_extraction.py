from feature_extraction import extract_features

file_path = "data/RAVDESS/Actor_01/03-01-05-01-01-01-01.wav"

features = extract_features(file_path)

print("Feature vector shape:", features.shape)
print("First 10 feature values:", features[:10])
