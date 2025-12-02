from flask import Flask, request, jsonify
import numpy as np

from .detect_anomaly import load_artifacts
from .config import BEAT_SIZE

app = Flask(__name__)

# Load model, scaler, threshold once at startup
model, scaler, threshold = load_artifacts()

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "ECG Autoencoder API running"})

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON:
    {
        "beat": [float, float, ..., float]   # length = BEAT_SIZE
    }
    """
    data = request.get_json(force=True)

    if "beat" not in data:
        return jsonify({"error": "Missing 'beat' in request body"}), 400

    beat = np.array(data["beat"], dtype=float)

    if beat.shape[0] != BEAT_SIZE:
        return jsonify({
            "error": f"Beat length must be {BEAT_SIZE}, got {beat.shape[0]}"
        }), 400

    # Scale using saved scaler
    beat_scaled = scaler.transform(beat.reshape(1, -1))

    # Reconstruction
    reconstructed = model.predict(beat_scaled)
    error = float(np.mean(np.square(beat_scaled - reconstructed)))

    is_anomaly = error > threshold

    return jsonify({
        "anomaly": bool(is_anomaly),
        "reconstruction_error": error,
        "threshold": float(threshold)
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
