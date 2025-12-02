import os
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

from .config import (
    BEAT_SIZE, TEST_SIZE, RANDOM_STATE, NORMAL_LABEL,
    SCALER_PATH, MODEL_DIR
)

def _create_model_dir():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)

def bandpass_filter(signal: np.ndarray,
                    lowcut: float = 0.5,
                    highcut: float = 40.0,
                    fs: float = 360.0,
                    order: int = 4) -> np.ndarray:
    """
    Apply Butterworth bandpass filter to a 1D ECG beat.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype="band")
    filtered = filtfilt(b, a, signal)
    return filtered

def preprocess_ecg(signals: np.ndarray,
                   labels: np.ndarray,
                   save_scaler: bool = True):
    """
    1. Bandpass filter each beat
    2. Standardize (mean=0, std=1)
    3. Train/test split
    4. Extract only normal beats for training autoencoder
    """
    # 1. Filter each beat
    filtered = np.apply_along_axis(bandpass_filter, 1, signals)

    # 2. Standardization
    scaler = StandardScaler()
    scaled = scaler.fit_transform(filtered)

    if save_scaler:
        _create_model_dir()
        joblib.dump(scaler, SCALER_PATH)
        print(f"[INFO] Scaler saved to {SCALER_PATH}")

    # 3. Train/test split
    x_train, x_test, y_train, y_test = train_test_split(
        scaled,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    # 4. Normal beats for training (label == NORMAL_LABEL)
    x_train_normals = x_train[y_train == NORMAL_LABEL]

    print(f"[INFO] Total beats: {len(signals)}")
    print(f"[INFO] Train beats: {len(x_train)}, Test beats: {len(x_test)}")
    print(f"[INFO] Normal beats in train (for AE): {len(x_train_normals)}")

    return x_train_normals, x_test, y_test
