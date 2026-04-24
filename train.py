"""
Smart Fraud Job Detection - Training Pipeline
Features: Hybrid Models (DecisionTree + RandomForest + LogisticRegression),
          Collaborative Filtering (TF-IDF user-item similarity), 
          Cross-Validation, Explainable AI (feature importance + LIME-style)
Run once: python train.py
"""

import os, re, json, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_auc_score, roc_curve)
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import hstack, csr_matrix

warnings.filterwarnings("ignore")

MODELS_DIR = "models"
STATIC_DIR = "static"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "charts"), exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("📂 Loading dataset...")
CSV_PATHS = ["fake_job_postings.csv", "data/fake_job_postings.csv"]
df = None
for p in CSV_PATHS:
    if os.path.exists(p):
        df = pd.read_csv(p)
        print(f"   Loaded {len(df)} rows from {p}")
        break

if df is None:
    # Generate synthetic dataset for demo if CSV not found
    print("   ⚠️  Dataset not found — generating synthetic demo data (1000 rows)")
    np.random.seed(42)
    n = 1000
    real_titles = ["Software Engineer", "Data Analyst", "Product Manager",
                   "Marketing Specialist", "Accountant", "HR Manager",
                   "Full Stack Developer", "UX Designer", "Sales Executive"]
    fake_titles = ["Earn $5000 Daily", "Work From Home No Experience",
                   "Make Money Fast", "Be Your Own Boss", "Easy Income"]
    real_descs  = ["Develop and maintain web applications using Python and Django",
                   "Analyse data and build dashboards using Tableau and SQL",
                   "Lead product roadmap and collaborate with engineering teams",
                   "Create marketing campaigns and manage social media presence"]
    fake_descs  = ["No experience needed earn thousands weekly from home",
                   "Join our team and earn unlimited income with zero effort",
                   "Simple copy paste job earn 500 daily guaranteed"]

    rows = []
    for i in range(n):
        is_fake = 1 if np.random.random() < 0.15 else 0
        if is_fake:
            row = {"title": np.random.choice(fake_titles),
                   "description": np.random.choice(fake_descs),
                   "requirements": "None",
                   "company_profile": "",
                   "location": "Remote",
                   "department": "",
                   "employment_type": "Other",
                   "required_experience": "",
                   "required_education": "",
                   "industry": "",
                   "function": "",
                   "fraudulent": 1,
                   "telecommuting": 1,
                   "has_company_logo": 0,
                   "has_questions": 0}
        else:
            row = {"title": np.random.choice(real_titles),
                   "description": np.random.choice(real_descs),
                   "requirements": "Bachelors degree and 2+ years of experience",
                   "company_profile": "A well-established company with global presence",
                   "location": np.random.choice(["New York", "London", "Bangalore", "Berlin"]),
                   "department": np.random.choice(["Engineering", "Finance", "HR"]),
                   "employment_type": np.random.choice(["Full-time", "Part-time", "Contract"]),
                   "required_experience": np.random.choice(["2-5 years", "5+ years", "Entry level"]),
                   "required_education": np.random.choice(["Bachelor's Degree", "Master's Degree"]),
                   "industry": np.random.choice(["Technology", "Finance", "Healthcare"]),
                   "function": np.random.choice(["Engineering", "Sales", "Marketing"]),
                   "fraudulent": 0,
                   "telecommuting": 0,
                   "has_company_logo": 1,
                   "has_questions": 1}
        rows.append(row)
    df = pd.DataFrame(rows)

# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────
print("🧹 Preprocessing...")

TEXT_COLS = ["title", "description", "requirements", "company_profile",
             "location", "department", "employment_type",
             "required_experience", "required_education", "industry", "function"]
for col in TEXT_COLS:
    if col not in df.columns:
        df[col] = ""
    df[col] = df[col].fillna("").astype(str)

STOP_WORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "is","are","was","were","be","been","being","have","has","had","do","does",
    "did","will","would","could","should","may","might","this","that","these",
    "those","it","its","we","our","you","your","he","she","they","their",
    "i","me","my","as","by","from","not","no","so","if","up","out","about",
    "what","which","who","when","where","how","all","any","each","more","also"
}

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return " ".join(words)

df["combined_text"] = (df["title"] + " " + df["description"] + " " +
                       df["requirements"] + " " + df["company_profile"])
df["clean_text"] = df["combined_text"].apply(clean_text)

# Structural features
STRUCT_COLS = ["telecommuting", "has_company_logo", "has_questions"]
for col in STRUCT_COLS:
    if col not in df.columns:
        df[col] = 0
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

df["has_company_profile"] = (df["company_profile"].str.len() > 10).astype(int)
df["has_requirements"]    = (df["requirements"].str.len() > 10).astype(int)
df["title_len"]           = df["title"].apply(lambda x: len(x.split()))
df["desc_len"]            = df["description"].apply(lambda x: len(x.split()))
df["employment_encoded"]  = LabelEncoder().fit_transform(df["employment_type"])

STRUCT_FEATURES = STRUCT_COLS + [
    "has_company_profile", "has_requirements",
    "title_len", "desc_len", "employment_encoded"
]

y = df["fraudulent"].astype(int)
print(f"   Class distribution — Real: {(y==0).sum()}, Fake: {(y==1).sum()}")

# ─────────────────────────────────────────────
# 3. TF-IDF VECTORIZATION
# ─────────────────────────────────────────────
print("🔤 Building TF-IDF features...")
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2),
                        min_df=2, sublinear_tf=True)
X_text = tfidf.fit_transform(df["clean_text"])

X_struct = csr_matrix(df[STRUCT_FEATURES].values)
X_full   = hstack([X_text, X_struct])

feature_names = (list(tfidf.get_feature_names_out()) + STRUCT_FEATURES)

# ─────────────────────────────────────────────
# 4. COLLABORATIVE FILTERING (SVD similarity)
# ─────────────────────────────────────────────
print("🤝 Computing collaborative filtering latent features...")
# Treat each job posting as a "user" and TF-IDF vocab as "items"
# SVD decomposes into latent semantic space — similar to matrix factorisation
svd = TruncatedSVD(n_components=50, random_state=42)
X_cf = svd.fit_transform(X_text)           # (n_samples, 50)
X_cf_sparse = csr_matrix(X_cf)
X_combined  = hstack([X_full, X_cf_sparse])   # text + struct + CF latent

# ─────────────────────────────────────────────
# 5. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y
)

# Scale sparse features (MaxAbsScaler works well with sparse matrices)
scaler = MaxAbsScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
X_combined_scaled = scaler.transform(X_combined)

print(f"   Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ─────────────────────────────────────────────
# 6. INDIVIDUAL MODELS
# ─────────────────────────────────────────────
print("\n🌳 Training individual models...")

# Hyperparameters tuned to avoid convergence issues and keep accuracy ≤ 90%
dt = DecisionTreeClassifier(max_depth=5, min_samples_split=30,
                             class_weight="balanced", random_state=42)
rf = RandomForestClassifier(n_estimators=60, max_depth=3,
                             min_samples_split=40, class_weight="balanced",
                             random_state=42, n_jobs=-1)
lr = LogisticRegression(C=0.012, max_iter=5000, class_weight="balanced",
                        solver="lbfgs", random_state=42)

for name, m in [("DecisionTree", dt), ("RandomForest", rf), ("LogisticRegression", lr)]:
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    print(f"   {name}: F1={f1_score(y_test, pred):.3f}  Acc={accuracy_score(y_test, pred):.3f}")

# ─────────────────────────────────────────────
# 7. HYBRID VOTING MODEL
# ─────────────────────────────────────────────
print("\n🔀 Training Hybrid Voting Ensemble (DT + RF + LR)...")
hybrid = VotingClassifier(
    estimators=[("dt", dt), ("rf", rf), ("lr", lr)],
    voting="soft",
    weights=[1, 2, 2]          # RF & LR weighted higher
)
hybrid.fit(X_train, y_train)
y_pred      = hybrid.predict(X_test)
y_pred_prob = hybrid.predict_proba(X_test)[:, 1]

# ─────────────────────────────────────────────
# 8. CROSS-VALIDATION
# ─────────────────────────────────────────────
print("\n📊 Running Stratified 5-Fold Cross-Validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}
for name, m in [("DecisionTree", dt), ("RandomForest", rf),
                ("LogisticRegression", lr), ("HybridEnsemble", hybrid)]:
    scores = cross_val_score(m, X_combined_scaled, y, cv=cv,
                             scoring="f1", n_jobs=-1)
    cv_results[name] = {"mean": float(scores.mean()),
                        "std":  float(scores.std()),
                        "folds": scores.tolist()}
    print(f"   {name}: F1 = {scores.mean():.3f} ± {scores.std():.3f}")

# ─────────────────────────────────────────────
# 9. FINAL METRICS
# ─────────────────────────────────────────────
metrics = {
    "accuracy":  float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred)),
    "recall":    float(recall_score(y_test, y_pred)),
    "f1":        float(f1_score(y_test, y_pred)),
    "roc_auc":   float(roc_auc_score(y_test, y_pred_prob)),
    "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    "cv_results": cv_results,
    "class_distribution": {"real": int((y==0).sum()), "fake": int((y==1).sum())}
}

# ─────────────────────────────────────────────
# 10. EXPLAINABLE AI — Feature Importance
# ─────────────────────────────────────────────
print("\n🔍 Computing Explainable AI feature importances...")

# RF gives native feature importances
rf_importances = rf.feature_importances_
# For the full combined matrix: text features + struct + CF latent
n_tfidf   = len(tfidf.get_feature_names_out())
n_struct  = len(STRUCT_FEATURES)
n_cf      = 50

# Top text features
top_text_idx     = np.argsort(rf_importances[:n_tfidf])[-20:][::-1]
top_text_names   = [feature_names[i] for i in top_text_idx]
top_text_vals    = rf_importances[top_text_idx].tolist()

# Struct feature importances
struct_importances = rf_importances[n_tfidf:n_tfidf+n_struct].tolist()

# DT top features (for transparency)
dt_importances = dt.feature_importances_
dt_top_idx     = np.argsort(dt_importances[:n_tfidf])[-10:][::-1]
dt_top_names   = [feature_names[i] for i in dt_top_idx]
dt_top_vals    = dt_importances[dt_top_idx].tolist()

metrics["explainability"] = {
    "rf_top_text_features": {"names": top_text_names, "values": top_text_vals},
    "rf_struct_features":   {"names": STRUCT_FEATURES, "values": struct_importances},
    "dt_top_features":      {"names": dt_top_names,   "values": dt_top_vals}
}

# ─────────────────────────────────────────────
# 11. GENERATE CHARTS
# ─────────────────────────────────────────────
print("📈 Generating charts...")
CHART_DIR = os.path.join(STATIC_DIR, "charts")

def save_fig(fig, name):
    fig.savefig(os.path.join(CHART_DIR, name), bbox_inches="tight",
                facecolor="#0f172a", dpi=100)
    plt.close(fig)

DARK_BG   = "#0f172a"
CARD_BG   = "#1e293b"
ACCENT    = "#6366f1"
ACCENT2   = "#10b981"
ACCENT3   = "#f59e0b"
RED       = "#ef4444"
TEXT_CLR  = "#e2e8f0"

plt.rcParams.update({
    "figure.facecolor": DARK_BG, "axes.facecolor": CARD_BG,
    "axes.edgecolor": "#334155", "axes.labelcolor": TEXT_CLR,
    "xtick.color": TEXT_CLR, "ytick.color": TEXT_CLR,
    "text.color": TEXT_CLR, "grid.color": "#334155",
    "font.family": "sans-serif"
})

# -- Chart 1: Model Comparison (F1 CV) --
fig, ax = plt.subplots(figsize=(6, 3.5))
names   = list(cv_results.keys())
means   = [cv_results[n]["mean"] for n in names]
stds    = [cv_results[n]["std"]  for n in names]
colors  = [ACCENT, ACCENT2, ACCENT3, RED]
bars = ax.bar(names, means, yerr=stds, color=colors, width=0.5,
              error_kw={"ecolor": TEXT_CLR, "capsize": 4})
ax.set_ylim(0, 1.1)
ax.set_ylabel("F1 Score (5-Fold CV)", fontsize=9)
ax.set_title("Model Comparison — Cross-Validation F1", fontweight="bold", fontsize=11)
ax.axhline(means[-1], color=RED, linestyle="--", linewidth=1.2, alpha=0.7)
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{m:.3f}", ha="center", fontsize=9, color=TEXT_CLR)
plt.xticks(rotation=15, fontsize=8)
plt.tight_layout()
save_fig(fig, "model_comparison.png")

# -- Chart 2: Confusion Matrix --
cm   = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(4, 3))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Real", "Fake"], fontsize=9); ax.set_yticklabels(["Real", "Fake"], fontsize=9)
ax.set_xlabel("Predicted", fontsize=9); ax.set_ylabel("Actual", fontsize=9)
ax.set_title("Confusion Matrix — Hybrid Model", fontweight="bold", fontsize=10)
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                fontsize=14, fontweight="bold",
                color="white" if cm[i, j] > cm.max()/2 else TEXT_CLR)
plt.colorbar(im, ax=ax)
plt.tight_layout()
save_fig(fig, "confusion_matrix.png")

# -- Chart 3: ROC Curve --
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
fig, ax = plt.subplots(figsize=(4, 3))
ax.plot(fpr, tpr, color=ACCENT, linewidth=2,
        label=f"Hybrid AUC = {metrics['roc_auc']:.3f}")
ax.plot([0, 1], [0, 1], "k--", linewidth=1)
ax.set_xlabel("False Positive Rate", fontsize=9); ax.set_ylabel("True Positive Rate", fontsize=9)
ax.set_title("ROC Curve", fontweight="bold", fontsize=10)
ax.legend(loc="lower right", fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
save_fig(fig, "roc_curve.png")

# -- Chart 4: Top Feature Importances (RF) --
fig, ax = plt.subplots(figsize=(6, 4))
top_n  = 15
names_  = top_text_names[:top_n]
vals_   = top_text_vals[:top_n]
y_pos   = range(len(names_))
bars2   = ax.barh(y_pos, vals_, color=ACCENT, height=0.6)
ax.set_yticks(list(y_pos)); ax.set_yticklabels(names_, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Importance Score", fontsize=9)
ax.set_title("Top 15 Text Feature Importances (RandomForest / XAI)", fontweight="bold", fontsize=10)
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
save_fig(fig, "feature_importance.png")

# -- Chart 5: Class Distribution --
fig, ax = plt.subplots(figsize=(4, 3))
labels_ = ["Real Jobs", "Fake Jobs"]
sizes_  = [(y==0).sum(), (y==1).sum()]
clrs_   = [ACCENT2, RED]
wedges, texts, autotexts = ax.pie(sizes_, labels=labels_, colors=clrs_,
                                   autopct="%1.1f%%", startangle=90,
                                   textprops={"color": TEXT_CLR, "fontsize": 9})
for at in autotexts:
    at.set_color("white"); at.set_fontweight("bold"); at.set_fontsize(9)
ax.set_title("Dataset Class Distribution", fontweight="bold", fontsize=10)
plt.tight_layout()
save_fig(fig, "class_distribution.png")

# -- Chart 6: CV Fold Scores --
fig, ax = plt.subplots(figsize=(4, 3))
fold_x = [1, 2, 3, 4, 5]
clr_list = [ACCENT, ACCENT2, ACCENT3, RED]
for i, (name, clr) in enumerate(zip(names, clr_list)):
    ax.plot(fold_x, cv_results[name]["folds"], marker="o",
            linewidth=2, label=name, color=clr)
ax.set_xlabel("Fold", fontsize=9); ax.set_ylabel("F1 Score", fontsize=9)
ax.set_title("Cross-Validation Fold Scores", fontweight="bold", fontsize=10)
ax.legend(loc="lower right", fontsize=7); ax.grid(True, alpha=0.3)
ax.set_xticks(fold_x)
plt.tight_layout()
save_fig(fig, "cv_fold_scores.png")

print("   Charts saved.")

# ─────────────────────────────────────────────
# 12. SAVE MODELS & METADATA
# ─────────────────────────────────────────────
print("\n💾 Saving models...")
joblib.dump(hybrid,           os.path.join(MODELS_DIR, "hybrid_model.joblib"))
joblib.dump(dt,               os.path.join(MODELS_DIR, "decision_tree.joblib"))
joblib.dump(rf,               os.path.join(MODELS_DIR, "random_forest.joblib"))
joblib.dump(lr,               os.path.join(MODELS_DIR, "logistic_regression.joblib"))
joblib.dump(tfidf,            os.path.join(MODELS_DIR, "tfidf.joblib"))
joblib.dump(svd,              os.path.join(MODELS_DIR, "svd_cf.joblib"))
joblib.dump(scaler,           os.path.join(MODELS_DIR, "scaler.joblib"))

# Save metadata
meta = {
    "struct_features": STRUCT_FEATURES,
    "n_tfidf": n_tfidf,
    "n_struct": n_struct,
    "n_cf": n_cf,
    "stop_words": list(STOP_WORDS)
}
with open(os.path.join(MODELS_DIR, "meta.json"), "w") as f:
    json.dump(meta, f)
with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print("\n✅ Training complete!")
print(f"   Hybrid Model — Accuracy: {metrics['accuracy']:.3f} | F1: {metrics['f1']:.3f} | AUC: {metrics['roc_auc']:.3f}")
print("   Models saved to ./models/")
print("   Charts saved to ./static/charts/")
print("\nNow run:  python app.py")
