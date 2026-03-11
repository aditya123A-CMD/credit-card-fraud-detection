from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# ── Load model files ──────────────────────────────────────
MODEL_PATH = 'fraud_model.pkl'
SCALER_PATH = 'scaler.pkl'
FEATURES_PATH = 'features.pkl'

model        = None
scaler       = None
feature_names = None

def load_model():
    global model, scaler, feature_names
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        with open(FEATURES_PATH, 'rb') as f:
            feature_names = pickle.load(f)
        print("✅  Model loaded successfully.")
    except Exception as e:
        print(f"⚠️  Could not load model: {e}  — running in demo mode.")

load_model()


# ── Routes ────────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        amount = float(data.get('amount', 0))
        hour   = float(data.get('hour',   0))
        v_vals = [float(data.get(f'v{i}', 0)) for i in range(1, 9)]

        features = np.array([[amount, hour] + v_vals])

        # ── Demo mode (no model files) ──
        if model is None or scaler is None:
            import random
            fraud_prob = random.uniform(0.01, 0.99)
            is_fraud   = fraud_prob > 0.5
            return jsonify({
                'result':                 'FRAUD' if is_fraud else 'LEGITIMATE',
                'fraud_probability':       round(fraud_prob * 100, 2),
                'legitimate_probability':  round((1 - fraud_prob) * 100, 2),
                'risk_level':              _risk(fraud_prob),
                'demo_mode':               True
            })

        # ── Real prediction ──
        features_scaled = scaler.transform(features)
        prediction      = model.predict(features_scaled)[0]
        probabilities   = model.predict_proba(features_scaled)[0]

        fraud_prob = float(probabilities[1])
        legit_prob = float(probabilities[0])

        return jsonify({
            'result':                 'FRAUD' if prediction == 1 else 'LEGITIMATE',
            'fraud_probability':       round(fraud_prob * 100, 2),
            'legitimate_probability':  round(legit_prob * 100, 2),
            'risk_level':              _risk(fraud_prob),
            'demo_mode':               False
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def _risk(p):
    if p >= 0.75: return 'CRITICAL'
    if p >= 0.50: return 'HIGH'
    if p >= 0.25: return 'MEDIUM'
    return 'LOW'


# ── Entry point ───────────────────────────────────────────
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)




