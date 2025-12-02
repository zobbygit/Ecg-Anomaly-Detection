ECG Anomaly Detection using Autoencoder
🫀 ECG Anomaly Detection (Autoencoder)

A deep learning–based unsupervised anomaly detection system that identifies abnormal ECG heartbeats using a reconstruction-error Autoencoder trained on real ECG5000 dataset.
Includes full data pipeline, training scripts, anomaly detector, and a real-time Flask API.

ECG-Autoencoder-Project/
│
├── data/
│   ├── ECG5000_TRAIN.txt      # (or ecg5000.csv)
│   ├── ECG5000_TEST.txt
│   ├── ecg_signals.npy        # generated
│   ├── ecg_labels.npy         # generated
│
├── models/
│   ├── ecg_autoencoder.h5     # trained model
│   ├── scaler.joblib          # StandardScaler
│   ├── threshold.npy          # anomaly threshold
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train_autoencoder.py
│   ├── detect_anomaly.py
│   ├── api.py
│   ├── prepare_ecg5000.py
│   ├── evaluate_model.py
│   └── test_api.py
│
├── requirements.txt
└── README.md

Dataset Setup (ECG5000)

Place any of these files inside the /data folder:

ECG5000_TRAIN.txt

ECG5000_TEST.txt

OR ecg5000.csv

Then generate NumPy files:

python -m src.prepare_ecg5000

rain using real normal ECG beats:

python -m src.train_autoencoder


This will save:

models/ecg_autoencoder.h5

models/scaler.joblib

models/threshold.npy

Test Reconstruction (Optional)

Plot original vs reconstructed ECG beats:

python -m src.plot_reconstruction

python -m src.api

{
  "status": "ok",
  "message": "ECG Autoencoder API running"
}


{
  "anomaly": true,
  "reconstruction_error": 0.0473,
  "threshold": 0.0387
}


python -m src.test_api

Technologies Used

Python 3.x

TensorFlow / Keras

NumPy / Pandas

Scikit-learn

Flask

Joblib

📘 How It Works (Concept)

Autoencoder is trained only on normal ECG beats

Learns the “normal heartbeat pattern”

During testing:

Reconstructs the beat

Computes reconstruction error

Compares with threshold

High error → Anomaly detected

This uses unsupervised anomaly detection with real cardiac data.
