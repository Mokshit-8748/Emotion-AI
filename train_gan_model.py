import argparse
import json
import os

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import DATASET_PATHS, MODEL_PATH_GAN
from src.gan_integration import prepare_gan_augmented_dataset
from src.train_model import train_emotion_model


DEFAULT_GAN_DATASETS = [
    path for path in DATASET_PATHS if os.path.basename(path).lower() != "iemocap"
]


def main():
    parser = argparse.ArgumentParser(description="Train the GAN-augmented SER classifier.")
    parser.add_argument("--datasets", default=",".join(DEFAULT_GAN_DATASETS))
    parser.add_argument("--augmentation_factor", type=float, default=1.0)
    parser.add_argument("--gan_epochs", type=int, default=300)
    parser.add_argument("--gan_batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.45)
    parser.add_argument("--output", default=MODEL_PATH_GAN)
    parser.add_argument("--quick_run", action="store_true")
    args = parser.parse_args()

    dataset_paths = [p.strip() for p in args.datasets.split(",") if p.strip()]
    gan_epochs = 10 if args.quick_run else args.gan_epochs
    clf_epochs = 3 if args.quick_run else args.epochs

    print("=" * 72)
    print("  EMOTION AI - GAN ENHANCED TRAINING")
    print("=" * 72)
    print(f"Datasets: {dataset_paths}")
    print(f"GAN epochs: {gan_epochs} | GAN batch size: {args.gan_batch_size}")
    print(f"Classifier epochs: {clf_epochs} | Batch size: {args.batch_size}")
    print(f"Augmentation factor: {args.augmentation_factor}")

    X_train, X_test, y_train, y_test, artifacts = prepare_gan_augmented_dataset(
        dataset_paths,
        use_gan=True,
        augmentation_factor=args.augmentation_factor,
        gan_epochs=gan_epochs,
        gan_batch_size=args.gan_batch_size,
        return_artifacts=True,
        verbose=1,
    )

    print(f"Augmented training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_test)}")
    print(f"Classes: {list(artifacts['label_encoder'].classes_)}")

    model, history, scaler, label_encoder = train_emotion_model(
        X_train,
        y_train,
        X_val=X_test,
        y_val=y_test,
        label_encoder=artifacts["label_encoder"],
        scaler=artifacts["scaler"],
        output_path=args.output,
        epochs=clf_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dropout=args.dropout,
        inputs_are_scaled=True,
        save_artifacts=True,
    )

    X_test_r = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    probabilities = model.predict(X_test_r, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    acc = accuracy_score(y_test, predictions)
    best_val_acc = max(history.history.get("val_accuracy", [acc]))

    print("\n" + "=" * 72)
    print(f"FINAL GAN MODEL ACCURACY: {acc * 100:.2f}%")
    print(f"BEST VALIDATION ACCURACY: {best_val_acc * 100:.2f}%")
    print("=" * 72)
    print(classification_report(y_test, predictions, target_names=label_encoder.classes_, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("[DONE] " + json.dumps({"accuracy": float(acc), "best_val_accuracy": float(best_val_acc)}))


if __name__ == "__main__":
    main()
