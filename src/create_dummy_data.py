import os
import numpy as np

# Where to save the files
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

os.makedirs(DATA_DIR, exist_ok=True)

N_SAMPLES = 2000     # number of heartbeats (you can change)
BEAT_SIZE = 200      # must match config.py

# Create random "ECG-like" data
signals = np.random.randn(N_SAMPLES, BEAT_SIZE).astype(np.float32)

# Create random labels: 0 = normal, 1 = abnormal
# Say 80% normal, 20% abnormal
labels = np.zeros(N_SAMPLES, dtype=np.int32)
labels[int(0.8 * N_SAMPLES):] = 1

# Shuffle them
idx = np.random.permutation(N_SAMPLES)
signals = signals[idx]
labels = labels[idx]

# Save to .npy files
signals_path = os.path.join(DATA_DIR, "ecg_signals.npy")
labels_path = os.path.join(DATA_DIR, "ecg_labels.npy")

np.save(signals_path, signals)
np.save(labels_path, labels)

print(f"Saved dummy signals to: {signals_path}, shape = {signals.shape}")
print(f"Saved dummy labels  to: {labels_path}, shape = {labels.shape}")
