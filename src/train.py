from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf
import yaml


PROCESSED_FILE = Path("data/processed/fashion_mnist_processed.npz")
MODELS_DIR = Path("models")
MODEL_FILE = MODELS_DIR / "model.h5"
HISTORY_FILE = MODELS_DIR / "history.csv"
PARAMS_FILE = Path("params.yaml")


def load_params() -> dict:
    with PARAMS_FILE.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)["train"]


def build_model(params: dict) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(params["dense_units"], activation="relu"),
            tf.keras.layers.Dropout(params["dropout_rate"]),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    params = load_params()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    data = np.load(PROCESSED_FILE)
    model = build_model(params)
    history = model.fit(
        data["x_train"],
        data["y_train"],
        validation_data=(data["x_val"], data["y_val"]),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
    )

    model.save(MODEL_FILE)
    pd.DataFrame(history.history).to_csv(HISTORY_FILE, index=False)

    print(f"Saved model to {MODEL_FILE}")
    print(f"Saved training history to {HISTORY_FILE}")


if __name__ == "__main__":
    main()
