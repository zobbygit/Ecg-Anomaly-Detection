import os
import numpy as np
from .config import DATA_SIGNAL_PATH, DATA_LABEL_PATH, BEAT_SIZE

def load_ecg_data(
    signal_path: str = DATA_SIGNAL_PATH,
    label_path: str = DATA_LABEL_PATH
):
    """
    Load ECG data from .npy files.
    ecg_signals.npy shape: (N, BEAT_SIZE)
    ecg_labels.npy  shape: (N,)
    """
    if not os.path.exists(signal_path):
        raise FileNotFoundError(
            f"ECG signals file not found at {signal_path}. "
            "Please save your beats as a numpy array of shape (N, BEAT_SIZE)."
        )

    if not os.path.exists(label_path):
        raise FileNotFoundError(
            f"ECG labels file not found at {label_path}. "
            "Please save your labels as a numpy array of shape (N,)."
        )

    signals = np.load(signal_path)
    labels = np.load(label_path)

    if signals.ndim != 2 or signals.shape[1] != BEAT_SIZE:
        raise ValueError(
            f"Expected signals of shape (N, {BEAT_SIZE}), got {signals.shape}"
        )
    if labels.ndim != 1 or labels.shape[0] != signals.shape[0]:
        raise ValueError(
            f"Labels must be shape (N,), got {labels.shape} for N={signals.shape[0]}"
        )

    return signals, labels
