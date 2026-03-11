import pandas as pd
import numpy as np
import joblib, os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score,
    f1_score, precision_score, recall_score
)

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("📂 Loading dataset...")
df = pd.read_csv("outputs/dataset.csv")

# ── FEATURES & TARGET ──
features = ["Amount","Hour","V1","V2","V3","V4","V5",
            "TransactionFreq","MerchantDistance","CardholderAge"]

# ── BALANCE CLASSES (oversample fraud) ──
print("⚖️  Balancing dataset...")
fraud_df = df[df.Class == 1]
legit_df = df[df.Class == 0]
fraud_os = fraud_df.sample(n=len(legit_df)//2, replace=True, random_state=42)
df_bal   = pd.concat([legit_df, fraud_os]).sample(frac=1, random_state=42)

X = df_bal[features]
y = df_bal["Class"]

# ── TRAIN / TEST SPLIT ──
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── SCALE FEATURES ──
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)   # fit + transform train
X_test_sc  = scaler.transform(X_test)         # only transform test!

print(f"✅ Train: {len(X_train):,} | Test: {len(X_test):,}")

# ── DEFINE MODELS ──
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# ── TRAIN ALL MODELS ──
results = {}
for name, model in models.items():
    print(f"\n🔄 Training {name}...")
    model.fit(X_train_sc, y_train)

    y_pred  = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:, 1]

    results[name] = {
        "model":     model,
        "y_pred":    y_pred,
        "y_proba":   y_proba,
        "accuracy":  accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "roc_auc":   roc_auc_score(y_test, y_proba),
    }
    r = results[name]
    print(f"  AUC={r['roc_auc']:.4f} | F1={r['f1']:.4f} | Recall={r['recall']:.4f}")

# ── FIND BEST MODEL ──
best_name = max(results, key=lambda k: results[k]["roc_auc"])
best_model = results[best_name]["model"]

# ── SAVE MODEL & ARTIFACTS ──
joblib.dump(best_model, "models/fraud_model.pkl")
joblib.dump(scaler,     "models/scaler.pkl")
joblib.dump(features,   "models/features.pkl")

print(f"\n🏆 Best Model: {best_name}")
print("💾 Model saved to models/")

# ── PRINT SUMMARY TABLE ──
print("\n{'='*55}")
print(f"{'Model':<25} {'AUC':>8} {'F1':>8} {'Recall':>8}")
print("-"*55)
for name, r in results.items():
    star = " ⭐" if name == best_name else ""
    print(f"{name+star:<25} {r['roc_auc']:>8.4f} {r['f1']:>8.4f} {r['recall']:>8.4f}")