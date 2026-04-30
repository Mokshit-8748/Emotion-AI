from dataset_builder import build_dataset

X, y = build_dataset("data/RAVDESS")

print("Total samples:", X.shape[0])
print("Feature shape:", X.shape[1])
print("First 5 labels:", y[:5])
