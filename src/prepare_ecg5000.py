import os
import numpy as np
import pandas as pd
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Try to detect common ECG5000 file names
CANDIDATE_FILES = [
    "ECG5000_TRAIN.txt",
    "ECG5000_TEST.txt",
    "ECG5000_TRAIN.csv",
    "ECG5000_TEST.csv",
    "ecg5000.csv",
    "ECG5000.csv",
]

def load_ecg_file(path):
    """
    Try to load an ECG5000 file that may be space-separated or comma-separated.
    """
    print(f"[INFO] Trying to read: {path}")
    # First try normal CSV (comma separated)
    df = pd.read_csv(path, header=None)
    print("[INFO] First read shape:", df.shape)

    # If only 1 column, likely wrong delimiter → try whitespace-separated
    if df.shape[1] == 1:
        print("[WARN] Only 1 column detected, retrying with whitespace separator...")
        df = pd.read_csv(path, header=None, sep=r"\s+")
        print("[INFO] After whitespace read, shape:", df.shape)

    return df

def main():
    # Find which ECG5000 files exist
    available = []
    for name in CANDIDATE_FILES:
        p = os.path.join(DATA_DIR, name)
        if os.path.exists(p):
            available.append(p)

    if not available:
        raise FileNotFoundError(
            f"No ECG5000 file found in {DATA_DIR}.\n"
            f"Put ECG5000_TRAIN.txt / ECG5000_TEST.txt or ecg5000.csv there."
        )

    print("[INFO] Found these ECG files:")
    for p in available:
        print("  -", os.path.basename(p))

    # If we have both TRAIN and TEST, load and concat
    train_paths = [p for p in available if "TRAIN" in os.path.basename(p).upper()]
    test_paths  = [p for p in available if "TEST" in os.path.basename(p).upper()]

    if train_paths and test_paths:
        print("[INFO] Using TRAIN + TEST files and concatenating them.")

        dfs = []
        for p in train_paths + test_paths:
            df_part = load_ecg_file(p)
            dfs.append(df_part)
        df = pd.concat(dfs, axis=0, ignore_index=True)
    else:
        # Just use the first available file
        print("[INFO] Using single file:", os.path.basename(available[0]))
        df = load_ecg_file(available[0])

    print("[INFO] Final combined DataFrame shape:", df.shape)

    # Assume:
    # - column 0 = label
    # - remaining columns = ECG time steps
    if df.shape[1] < 2:
        raise ValueError(
            f"Expected at least 2 columns (label + features), but got {df.shape[1]}.\n"
            f"Open the file in a text editor and check the format."
        )

    labels_raw = df.iloc[:, 0].values
    signals = df.iloc[:, 1:].values.astype(np.float32)

    print("[INFO] Signals shape:", signals.shape)
    print("[INFO] First row (truncated):", signals[0, :10])
    print("[INFO] Sample raw labels:", labels_raw[:10])

    # Map multi-class labels → binary: 0 = normal, 1 = abnormal
    from collections import Counter
    counts = Counter(labels_raw)
    print("[INFO] Label counts:", counts)

    # Choose the most frequent class as "normal"
    normal_class = counts.most_common(1)[0][0]
    print(f"[INFO] Treating class {normal_class} as NORMAL (0), others as ABNORMAL (1).")

    labels_binary = (labels_raw != normal_class).astype(np.int32)

    # Save as .npy
    signals_path = os.path.join(DATA_DIR, "ecg_signals.npy")
    labels_path = os.path.join(DATA_DIR, "ecg_labels.npy")

    np.save(signals_path, signals)
    np.save(labels_path, labels_binary)

    print(f"[INFO] Saved signals to: {signals_path}, shape = {signals.shape}")
    print(f"[INFO] Saved labels  to: {labels_path}, shape = {labels_binary.shape}")
    print("[INFO] Done. Now run: python -m src.train_autoencoder")

if __name__ == "__main__":
    main()
