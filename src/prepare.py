from pathlib import Path
import os

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from tensorflow.keras.datasets import fashion_mnist


RAW_DIR = Path("data/raw")
RAW_FILE = RAW_DIR / "fashion_mnist_raw.npz"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    np.savez_compressed(
        RAW_FILE,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )

    print(f"Saved raw Fashion-MNIST arrays to {RAW_FILE}")
    print(f"Train images: {x_train.shape}, test images: {x_test.shape}")


if __name__ == "__main__":
    main()
