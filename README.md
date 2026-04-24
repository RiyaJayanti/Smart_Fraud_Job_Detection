# Smart Fraud Job Detection System

A **Machine Learning system** that detects fraudulent job postings using a Hybrid Ensemble model, Collaborative Filtering, Cross-Validation, and Explainable AI.

---

## Overview

Fake job postings are a serious issue in online recruitment platforms.
This project builds an intelligent system to **identify fraudulent job listings in real-time** and provide **explainable predictions** using modern ML techniques.

---

## Preview

*(Add your screenshot here)*

```md id="img1"
![App Screenshot](screenshot.png)
```

---

## Key Features

* 🔹 Hybrid Ensemble Model (Decision Tree + Random Forest + Logistic Regression)
* 🔹 Collaborative Filtering using SVD
* 🔹 Stratified 5-Fold Cross Validation
* 🔹 Explainable AI (Feature importance + keyword insights)
* 🔹 Interactive Flask Web Application
* 🔹 Auto-generated performance charts

---

## Results & Insights

* Accuracy: **90.2%**
* Precision: **32.1%**
* Recall: **93.1%**
* F1 Score: **0.478**
* AUC-ROC: **0.96**

### Key Insight

The dataset is **highly imbalanced (~5% fraudulent jobs)**.

The model is optimized for **high recall**, ensuring most fraudulent jobs are detected —
which is critical in fraud detection systems.

This leads to lower precision (more false positives), but reduces the risk of missing actual fraud.

---

## 📁 Project Structure

```bash id="tree1"
fraud_detector/
├── train.py
├── app.py
├── requirements.txt
├── templates/
│   └── index.html
├── static/
│   └── charts/
└── models/
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

## How It Works

### 🔹 Data Processing

* Text converted using **TF-IDF vectorization**
* Dimensionality reduction using **Truncated SVD**

### 🔹 Model Training

* Decision Tree
* Random Forest (higher voting weight)
* Logistic Regression (higher voting weight)
* Combined using **Soft Voting Ensemble**

### 🔹 Validation

* Stratified 5-Fold Cross Validation ensures reliable performance

### 🔹 Explainable AI

* Feature importance using Random Forest
* Decision Tree insights
* Keyword-level explanation for predictions

---

## Quick Start

### 1. Install Dependencies

```bash id="cmd1"
pip install -r requirements.txt
```

### 2. Add Dataset

Place `fake_job_postings.csv` in the root folder.

Dataset source: Kaggle – *Real or Fake Job Posting Prediction*

---

### 3. Train the Model (Run Once)

```bash id="cmd2"
python train.py
```

✔ Trains models
✔ Saves them in `/models`
✔ Generates charts

---

### 4. Run the Web App

```bash id="cmd3"
python app.py
```

🌐 Open: http://localhost:5000

---

## Web Application Features

* **Predict** → Analyze job postings instantly
* **Dashboard** → View model performance
* **Charts** → Visual insights
* **About** → System architecture

---

## Model Storage

All trained models are saved as `.joblib` files.
No retraining required unless dataset changes.

---

## Tech Stack

* Python
* Scikit-learn
* Flask
* Pandas / NumPy
* Matplotlib

---

## Future Improvements

* Deploy on cloud (AWS / Render)
* Improve precision using threshold tuning / SMOTE
* Enhance UI/UX
* Integrate deep learning models

---

## 👩‍💻 Author

**Riya Jayanti**
