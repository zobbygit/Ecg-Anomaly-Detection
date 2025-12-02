import os
import numpy as np
import joblib
import matplotlib.pyplot as plt
import tensorflow as tf

from .config import DATA_SIGNAL_PATH, DATA_LABEL_PATH
from .detect_anomaly import load_artifacts

def main():
    # Load data
    signals = np.load(DATA_SIGNAL_PATH)
    labels  = np.load(DATA_LABEL_PATH)

    # Load model, scaler, threshold
    model, scaler, threshold = load_artifacts()

    # Pick one normal and one abnormal beat
    normal_idx = np.where(labels == 0)[0][0]
    abnormal_idx = np.where(labels == 1)[0][0]

    for idx, title in [(normal_idx, "Normal Beat"), (abnormal_idx, "Abnormal Beat")]:
        beat_raw = signals[idx]
        beat_scaled = scaler.transform(beat_raw.reshape(1, -1))
        reconstructed = model.predict(beat_scaled)[0]

        plt.figure()
        plt.plot(beat_scaled[0], label="Original (scaled)")
        plt.plot(reconstructed, linestyle="--", label="Reconstructed")
        plt.title(f"{title} (index {idx})")
        plt.xlabel("Time step")
        plt.ylabel("Amplitude (scaled)")
        plt.legend()
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    main()
