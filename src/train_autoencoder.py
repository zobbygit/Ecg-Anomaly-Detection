import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from .config import (
    BEAT_SIZE, EPOCHS, BATCH_SIZE, LEARNING_RATE,
    MODEL_PATH, THRESHOLD_PATH, MODEL_DIR
)
from .data_loader import load_ecg_data
from .preprocessing import preprocess_ecg
from .model import build_autoencoder

def train():
    # 1. Load raw beats + labels
    signals, labels = load_ecg_data()

    # 2. Preprocess (filter, scale, split)
    x_train_normals, x_test, y_test = preprocess_ecg(signals, labels)

    # 3. Build autoencoder
    model = build_autoencoder(input_dim=BEAT_SIZE)
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss="mse"
    )

    model.summary()

    # 4. Train model (normal beats only)
    history = model.fit(
        x_train_normals,
        x_train_normals,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        shuffle=True
    )

    # 5. Save model
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)

    model.save(MODEL_PATH)
    print(f"[INFO] Model saved to {MODEL_PATH}")

    # 6. Compute reconstruction error on normal training beats
    reconstructed = model.predict(x_train_normals)
    train_errors = np.mean(
        np.square(x_train_normals - reconstructed),
        axis=1
    )

    # Threshold = mean + 3 * std
    threshold = train_errors.mean() + 3 * train_errors.std()
    np.save(THRESHOLD_PATH, threshold)
    print(f"[INFO] Threshold saved to {THRESHOLD_PATH}: {threshold:.6f}")

    # Simple evaluation hint
    print("[INFO] Training complete. Run evaluate/detect scripts for full metrics.")

if __name__ == "__main__":
    train()
