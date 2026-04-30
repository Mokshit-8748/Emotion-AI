from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from src.dataset_builder import build_dataset, build_multi_dataset


def prepare_data(dataset_path):
    """
    Accepts either:
      - a single string:  "data/RAVDESS"
      - a list of paths:  ["data/RAVDESS", "data/CREMA-D", "data/TESS"]
    """
    if isinstance(dataset_path, list):
        X, y = build_multi_dataset(dataset_path)
    else:
        X, y = build_dataset(dataset_path)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    return X_train, X_test, y_train, y_test, label_encoder
