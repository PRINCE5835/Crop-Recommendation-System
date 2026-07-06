import os
import sys
import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(PROJECT_DIR)

MODEL_PATH = os.path.join(PROJECT_DIR, "model", "crop_model.pkl")
SCALER_PATH = os.path.join(PROJECT_DIR, "model", "crop_scaler.pkl")

app = Flask(__name__)
CORS(app)

model = None
scaler = None

def load_model():
    global model, scaler
    if not os.path.exists(MODEL_PATH):
        return False, f"Model file not found at {MODEL_PATH}. Run model/train_model.py first."
    if not os.path.exists(SCALER_PATH):
        return False, f"Scaler file not found at {SCALER_PATH}. Run model/train_model.py first."
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return True, "Model loaded successfully."

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Crop Recommendation API is running."})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        loaded, msg = load_model()
        if not loaded:
            return jsonify({"error": msg}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON data received."}), 400

    required = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        features = np.array([[data["N"], data["P"], data["K"],
                              data["temperature"], data["humidity"],
                              data["ph"], data["rainfall"]]], dtype=float)
    except (ValueError, TypeError):
        return jsonify({"error": "All fields must be numeric."}), 400

    scaled = scaler.transform(features)
    prediction = model.predict(scaled)[0]
    probabilities = model.predict_proba(scaled)[0]
    top3_idx = np.argsort(probabilities)[-3:][::-1]
    top3 = [{"crop": model.classes_[i], "confidence": round(probabilities[i] * 100, 2)}
            for i in top3_idx]

    return jsonify({
        "recommended_crop": prediction,
        "top_3_crops": top3
    })

if __name__ == "__main__":
    loaded, msg = load_model()
    if not loaded:
        print(f"[ERROR] {msg}")
        sys.exit(1)
    print("[INFO] Starting Flask server on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
