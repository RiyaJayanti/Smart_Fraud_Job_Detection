"""
Smart Fraud Job Detection - Flask Web Application
Run: python app.py
"""

import os, re, json
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify
from scipy.sparse import hstack, csr_matrix

app = Flask(__name__)

MODELS_DIR = "models"
REQUIRED   = ["hybrid_model.joblib", "tfidf.joblib", "svd_cf.joblib",
               "scaler.joblib", "meta.json", "metrics.json"]

# ── Load artefacts ──────────────────────────────────────────────────────────
def load_models():
    missing = [f for f in REQUIRED if not os.path.exists(os.path.join(MODELS_DIR, f))]
    if missing:
        return None, None, None, None, None, None
    hybrid = joblib.load(os.path.join(MODELS_DIR, "hybrid_model.joblib"))
    tfidf  = joblib.load(os.path.join(MODELS_DIR, "tfidf.joblib"))
    svd    = joblib.load(os.path.join(MODELS_DIR, "svd_cf.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    with open(os.path.join(MODELS_DIR, "meta.json"))    as f: meta    = json.load(f)
    with open(os.path.join(MODELS_DIR, "metrics.json")) as f: metrics = json.load(f)
    return hybrid, tfidf, svd, scaler, meta, metrics

HYBRID, TFIDF, SVD, SCALER, META, METRICS = load_models()
STOP_WORDS = set(META["stop_words"]) if META else set()

# ── Preprocessing ─────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return " ".join(words)

def build_features(data: dict):
    combined = (data.get("title","") + " " + data.get("description","") + " " +
                data.get("requirements","") + " " + data.get("company_profile",""))
    clean = clean_text(combined)
    X_text   = TFIDF.transform([clean])
    X_cf     = csr_matrix(SVD.transform(X_text))
    struct   = [
        int(data.get("telecommuting", 0)),
        int(data.get("has_company_logo", 0)),
        int(data.get("has_questions", 0)),
        1 if len(data.get("company_profile","")) > 10 else 0,
        1 if len(data.get("requirements",""))    > 10 else 0,
        len(data.get("title","").split()),
        len(data.get("description","").split()),
        0   # employment_encoded placeholder
    ]
    X_struct = csr_matrix(np.array(struct).reshape(1, -1))
    X_full   = hstack([X_text, X_struct, X_cf])
    X_full   = SCALER.transform(X_full)
    return X_full, clean

# ── XAI helper ────────────────────────────────────────────────────────────
def explain_prediction(clean_text_str: str, prob: float):
    """Return top contributing words for this prediction."""
    rf = HYBRID.estimators_[1]          # RandomForest is index 1
    tfidf_vec = TFIDF.transform([clean_text_str])
    importance = rf.feature_importances_[:TFIDF.get_feature_names_out().shape[0]]
    word_scores = {}
    cx = tfidf_vec.tocsr()
    for idx in cx.indices:
        word = TFIDF.get_feature_names_out()[idx]
        score = float(cx[0, idx]) * float(importance[idx])
        word_scores[word] = score
    top = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    return [{"word": w, "score": round(s * 1000, 4)} for w, s in top if s > 0]

# ── Routes ────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    models_ready = HYBRID is not None
    return render_template("index.html", models_ready=models_ready,
                           metrics=METRICS if METRICS else {})

@app.route("/predict", methods=["POST"])
def predict():
    if HYBRID is None:
        return jsonify({"error": "Models not trained yet. Run train.py first."}), 503

    data = request.get_json(force=True)
    try:
        X, clean = build_features(data)
        prob     = float(HYBRID.predict_proba(X)[0][1])
        pred     = 1 if prob >= 0.5 else 0

        # Individual model predictions
        dt = HYBRID.estimators_[0]
        rf = HYBRID.estimators_[1]
        lr = HYBRID.estimators_[2]
        individual = {
            "DecisionTree":        round(float(dt.predict_proba(X)[0][1]), 3),
            "RandomForest":        round(float(rf.predict_proba(X)[0][1]), 3),
            "LogisticRegression":  round(float(lr.predict_proba(X)[0][1]), 3)
        }

        explanation = explain_prediction(clean, prob)

        risk_label = ("🔴 HIGH RISK — Likely Fake Job"   if prob > 0.7 else
                      "🟡 MEDIUM RISK — Suspicious"      if prob > 0.4 else
                      "🟢 LOW RISK — Likely Legitimate")

        return jsonify({
            "prediction":   pred,
            "probability":  round(prob, 4),
            "risk_label":   risk_label,
            "individual":   individual,
            "explanation":  explanation
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/metrics")
def get_metrics():
    if METRICS is None:
        return jsonify({"error": "No metrics found. Run train.py first."}), 503
    return jsonify(METRICS)

if __name__ == "__main__":
    if HYBRID is None:
        print("⚠️  Models not found. Please run:  python train.py")
    else:
        print("✅ Models loaded. Starting server...")
    app.run(debug=False, host="0.0.0.0", port=5000)
