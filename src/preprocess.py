from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split


RAW_FILE = Path("data/raw/fashion_mnist_raw.npz")
PROCESSED_DIR = Path("data/processed")
PROCESSED_FILE = PROCESSED_DIR / "fashion_mnist_processed.npz"
PARAMS_FILE = Path("params.yaml")


def load_params() -> dict:
    with PARAMS_FILE.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)["preprocess"]


def normalize(images: np.ndarray) -> np.ndarray:
    return (images.astype("float32") - 127.5) / 127.5


def main() -> None:
    params = load_params()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw = np.load(RAW_FILE)
    x_train = normalize(raw["x_train"])
    x_test = normalize(raw["x_test"])
    y_train = raw["y_train"]
    y_test = raw["y_test"]

    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=params["validation_size"],
        random_state=params["seed"],
        stratify=y_train,
    )

    np.savez_compressed(
        PROCESSED_FILE,
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        x_test=x_test,
        y_test=y_test,
    )

    print(f"Saved processed arrays to {PROCESSED_FILE}")
    print(f"Train: {x_train.shape}, val: {x_val.shape}, test: {x_test.shape}")


if __name__ == "__main__":
    main()
