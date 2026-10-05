import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


PROCESSED_FILE = Path("data/processed/fashion_mnist_processed.npz")
MODEL_FILE = Path("models/model.h5")
METRICS_FILE = Path("metrics.json")
CONFUSION_MATRIX_FILE = Path("models/confusion_matrix.png")
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def main() -> None:
    data = np.load(PROCESSED_FILE)
    model = tf.keras.models.load_model(MODEL_FILE)

    test_loss, test_accuracy = model.evaluate(
        data["x_test"],
        data["y_test"],
        verbose=0,
    )
    predictions = model.predict(data["x_test"], verbose=0).argmax(axis=1)
    matrix = confusion_matrix(data["y_test"], predictions)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES,
    )
    fig, ax = plt.subplots(figsize=(10, 10))
    display.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    fig.tight_layout()
    CONFUSION_MATRIX_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CONFUSION_MATRIX_FILE, dpi=150)
    plt.close(fig)

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
    }
    with METRICS_FILE.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
        file.write("\n")

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Saved metrics to {METRICS_FILE}")
    print(f"Saved confusion matrix to {CONFUSION_MATRIX_FILE}")


if __name__ == "__main__":
    main()
