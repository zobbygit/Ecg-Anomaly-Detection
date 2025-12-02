import numpy as np
import joblib
import tensorflow as tf
import os

from .config import (
    BEAT_SIZE,
    MODEL_PATH,
    SCALER_PATH,
    THRESHOLD_PATH,
    NORMAL_LABEL
)

def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train first.")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found at {SCALER_PATH}. Train first.")
    if not os.path.exists(THRESHOLD_PATH):
        raise FileNotFoundError(f"Threshold not found at {THRESHOLD_PATH}. Train first.")

    # Important: compile=False to avoid Keras deserializing 'mse' metric
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    threshold = float(np.load(THRESHOLD_PATH))

    return model, scaler, threshold

def detect_on_beats(beats: np.ndarray):
    """
    beats: numpy array of shape (N, BEAT_SIZE), raw unscaled beats.
    Returns:
        anomalies: bool array (True = anomaly)
        errors: reconstruction error for each beat
    """
    if beats.ndim == 1:
        beats = beats.reshape(1, -1)

    if beats.shape[1] != BEAT_SIZE:
        raise ValueError(f"Expected beats of length {BEAT_SIZE}, got {beats.shape[1]}.")

    model, scaler, threshold = load_artifacts()

    # Apply same scaling as training
    beats_scaled = scaler.transform(beats)

    # Reconstruction
    reconstructed = model.predict(beats_scaled)
    errors = np.mean(np.square(beats_scaled - reconstructed), axis=1)

    anomalies = errors > threshold
    return anomalies, errors, threshold

if __name__ == "__main__":
    # Example usage (dummy beat)
    dummy_beat = np.random.randn(BEAT_SIZE)
    anomalies, errors, threshold = detect_on_beats(dummy_beat)
    print("Errors:", errors)
    print("Threshold:", threshold)
    print("Is anomaly:", anomalies)
