from data_preparation import prepare_data

X_train, X_test, y_train, y_test, le = prepare_data("data/RAVDESS")

print("Train samples:", X_train.shape)
print("Test samples:", X_test.shape)
print("Encoded labels:", le.classes_)
