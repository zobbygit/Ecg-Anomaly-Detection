import os
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from .config import (
    DATA_SIGNAL_PATH,
    DATA_LABEL_PATH,
    TEST_SIZE,
    RANDOM_STATE,
)
from .detect_anomaly import load_artifacts

def main():
    # 1) Load raw signals + labels
    signals = np.load(DATA_SIGNAL_PATH)
    labels  = np.load(DATA_LABEL_PATH)

    print("[INFO] Loaded signals:", signals.shape)
    print("[INFO] Loaded labels :", labels.shape)

    # 2) Load trained model, scaler, threshold
    model, scaler, threshold = load_artifacts()

    # 3) Apply same scaler as training
    signals_scaled = scaler.transform(signals)

    # 4) Train/test split (same seed as training)
    _, x_test, _, y_test = train_test_split(
        signals_scaled,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    print("[INFO] Test set shape:", x_test.shape)

    # 5) Reconstruct test beats
    reconstructed = model.predict(x_test)
    errors = np.mean(np.square(x_test - reconstructed), axis=1)

    # 6) Anomaly decision using threshold
    y_pred = (errors > threshold).astype(int)

    print("[INFO] Threshold used:", threshold)
    print("[INFO] Reconstruction error stats:",
          "min =", errors.min(), "max =", errors.max(), "mean =", errors.mean())

    # 7) Metrics
    print("\nConfusion Matrix (rows = true, cols = pred):")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

if __name__ == "__main__":
    main()
