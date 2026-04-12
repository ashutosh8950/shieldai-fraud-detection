"""
train_model.py — Train and save the fraud detection ML model.
Run once: python train_model.py
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score, accuracy_score
)
from sklearn.pipeline import Pipeline
import joblib
import json

# ── Reproducibility ─────────────────────────────────────────────────────────
np.random.seed(42)

# ── 1. Synthetic dataset generation (pure numpy — no pandas needed) ──────────
FEATURES = [
    "amount", "hour", "day_of_week", "distance_from_home",
    "merchant_risk", "prev_txn_gap_hrs", "is_online",
    "num_daily_txns", "velocity_score",
]

def generate_dataset(n=10_000):
    """Generate a realistic synthetic transaction dataset using numpy only."""
    n_legit = int(n * 0.97)   # 97 % legitimate
    n_fraud = n - n_legit     # 3 % fraudulent

    # --- Legitimate transactions ---
    legit = np.column_stack([
        np.random.lognormal(4.0, 1.2, n_legit),            # amount
        np.random.randint(6, 23, n_legit).astype(float),   # hour
        np.random.randint(0, 7,  n_legit).astype(float),   # day_of_week
        np.random.exponential(10,  n_legit),                # distance_from_home
        np.random.beta(2, 8, n_legit),                      # merchant_risk
        np.random.exponential(24,  n_legit),                # prev_txn_gap_hrs
        np.random.binomial(1, 0.4, n_legit).astype(float), # is_online
        np.random.poisson(3, n_legit).astype(float),        # num_daily_txns
        np.random.beta(2, 8, n_legit),                      # velocity_score
        np.zeros(n_legit),                                   # label = 0
    ])

    # --- Fraudulent transactions (higher amounts, odd hours, etc.) ---
    fraud_hours = np.random.choice(
        list(range(0, 5)) + list(range(23, 24)), n_fraud
    ).astype(float)
    fraud = np.column_stack([
        np.random.lognormal(6.0, 1.5, n_fraud),            # amount
        fraud_hours,                                         # hour
        np.random.randint(0, 7, n_fraud).astype(float),    # day_of_week
        np.random.exponential(80, n_fraud),                 # distance_from_home
        np.random.beta(8, 2, n_fraud),                      # merchant_risk
        np.random.exponential(1, n_fraud),                  # prev_txn_gap_hrs
        np.random.binomial(1, 0.8, n_fraud).astype(float), # is_online
        np.random.poisson(10, n_fraud).astype(float),       # num_daily_txns
        np.random.beta(8, 2, n_fraud),                      # velocity_score
        np.ones(n_fraud),                                    # label = 1
    ])

    data = np.vstack([legit, fraud])
    # Shuffle
    idx = np.random.permutation(len(data))
    return data[idx]

print("🔄  Generating synthetic dataset …")
data = generate_dataset(10_000)
X = data[:, :-1]   # all feature columns
y = data[:, -1]    # label column

fraud_rate = y.mean() * 100
print(f"✅  Dataset: {len(data)} rows  |  fraud rate: {fraud_rate:.1f}%")

# ── 2. Train / test split ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 3. Build pipeline ────────────────────────────────────────────────────────
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf",    RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )),
])

print("🔄  Training Random Forest …")
model.fit(X_train, y_train)

# ── 4. Evaluation ────────────────────────────────────────────────────────────
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"\n📊  Accuracy : {acc*100:.2f}%")
print(f"📊  ROC-AUC  : {auc*100:.2f}%")
print("\n", classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

# ── 5. Save artefacts ────────────────────────────────────────────────────────
joblib.dump(model, "model.pkl")

meta = {
    "features":      FEATURES,
    "accuracy":      round(acc, 4),
    "roc_auc":       round(auc, 4),
    "model_type":    "RandomForestClassifier",
    "training_rows": len(X_train),
}
with open("model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\n✅  model.pkl and model_meta.json saved successfully!")
