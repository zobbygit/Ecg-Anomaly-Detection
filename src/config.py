import os

# General configuration
BEAT_SIZE = 140               # number of samples per ECG beat
TEST_SIZE = 0.2               # 20% test split
RANDOM_STATE = 42
NORMAL_LABEL = 0              # label for normal beats

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_SIGNAL_PATH = os.path.join(BASE_DIR, "data", "ecg_signals.npy")
DATA_LABEL_PATH  = os.path.join(BASE_DIR, "data", "ecg_labels.npy")

MODEL_DIR        = os.path.join(BASE_DIR, "models")
MODEL_PATH       = os.path.join(MODEL_DIR, "ecg_autoencoder.h5")
SCALER_PATH      = os.path.join(MODEL_DIR, "scaler.joblib")
THRESHOLD_PATH   = os.path.join(MODEL_DIR, "threshold.npy")

# Training config
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
VALIDATION_SPLIT = 0.2
