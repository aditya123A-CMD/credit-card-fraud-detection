import joblib, pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load saved model and data
model    = joblib.load("models/fraud_model.pkl")
scaler   = joblib.load("models/scaler.pkl")
features = joblib.load("models/features.pkl")
df       = pd.read_csv("outputs/dataset.csv")

X = df[features]
y = df["Class"]
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test_sc = scaler.transform(X_test)

y_pred  = model.predict(X_test_sc)
y_proba = model.predict_proba(X_test_sc)[:, 1]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)
axes[0].plot(fpr, tpr, color="#3498db", lw=2, label=f"AUC = {auc:.3f}")
axes[0].plot([0,1], [0,1], "k--", alpha=0.4)
axes[0].set_title("ROC Curve", fontweight="bold")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend()

# 2. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1],
           xticklabels=["Legit","Fraud"], yticklabels=["Legit","Fraud"])
axes[1].set_title("Confusion Matrix", fontweight="bold")
axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")

# 3. Feature Importance (if model supports it)
if hasattr(model, "feature_importances_"):
    imp = pd.Series(model.feature_importances_, index=features).sort_values()
    imp.plot(kind="barh", ax=axes[2], color="#3498db")
    axes[2].set_title("Feature Importance", fontweight="bold")

plt.tight_layout()
plt.savefig("outputs/evaluation.png", dpi=150)
plt.show()
print("✅ Evaluation charts saved!")