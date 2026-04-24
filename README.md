# 🕵️ Smart Fraud Job Detection System

A complete ML system to detect fraudulent job postings using a Hybrid Ensemble,
Collaborative Filtering, Cross-Validation, and Explainable AI.

---

## 📁 Project Structure
```
fraud_detector/
├── train.py             ← Full ML training pipeline (run once)
├── app.py               ← Flask web application
├── requirements.txt     ← Python dependencies
├── templates/
│   └── index.html       ← Frontend UI
├── static/
│   └── charts/          ← Auto-generated visualisation charts
└── models/              ← Saved models (created by train.py)
    ├── hybrid_model.joblib
    ├── decision_tree.joblib
    ├── random_forest.joblib
    ├── logistic_regression.joblib
    ├── tfidf.joblib
    ├── svd_cf.joblib
    ├── meta.json
    └── metrics.json
```

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your dataset
Place `fake_job_postings.csv` in the project root folder.
*(Download from Kaggle: shivamb/real-or-fake-fake-jobposting-prediction)*

If the CSV is not found, the system generates synthetic demo data automatically.

### 3. Train models (ONE TIME ONLY)
```bash
python train.py
```
This will:
- Preprocess and vectorize the data (TF-IDF)
- Compute collaborative filtering features (SVD)
- Train 3 models: Decision Tree, Random Forest, Logistic Regression
- Build the Hybrid Voting Ensemble
- Run 5-Fold Stratified Cross-Validation on all models
- Generate Explainable AI feature importances
- Save all models + metrics to `models/`
- Generate 6 charts to `static/charts/`

### 4. Run the web app
```bash
python app.py
```
Open: http://localhost:5000

---

## 🧠 ML Features

### Hybrid Ensemble Model
- **Decision Tree** (max_depth=15, balanced classes)
- **Random Forest** (150 trees, balanced classes) — 2× voting weight
- **Logistic Regression** (L2, balanced classes) — 2× voting weight
- Combined via **Soft Voting** for probability-averaged predictions

### Collaborative Filtering
- TF-IDF matrix decomposed via **Truncated SVD** (50 components)
- Captures latent semantic similarity across job postings
- CF features appended to the main feature matrix

### Cross-Validation
- **Stratified 5-Fold CV** preserves class distribution per fold
- Reported for all 4 models: mean F1 ± std

### Explainable AI (XAI)
- Global: RandomForest feature importances (top 15 text features)
- Global: Decision Tree top contributing features
- Per-prediction: TF-IDF weight × RF importance → key words

### No Re-training Needed
Models are saved as `.joblib` files. The app loads them at startup.
Only run `train.py` again if you change the dataset or model config.

---

## 🌐 Web Interface
- **Predict tab** — Analyse any job posting in real-time
- **Dashboard tab** — Model metrics (Accuracy, Precision, Recall, F1, AUC, CV)
- **Charts tab** — All 6 visualisation charts
- **About tab** — System architecture documentation
