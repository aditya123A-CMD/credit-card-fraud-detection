import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("outputs/dataset.csv")

print("=== DATASET INFO ===")
print(df.info())
print("\n=== FIRST 5 ROWS ===")
print(df.head())
print("\n=== FRAUD vs LEGIT ===")
print(df["Class"].value_counts())
print("\n=== STATISTICS ===")
print(df.describe())

# Create EDA dashboard
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Fraud Detection - EDA Dashboard", fontsize=16)

# 1. Class distribution
counts = df["Class"].value_counts()
axes[0,0].bar(["Legitimate", "Fraud"], counts.values,
            color=["#2ecc71", "#e74c3c"])
axes[0,0].set_title("Class Distribution")

# 2. Amount by class
axes[0,1].hist(df[df.Class==0]["Amount"], bins=50, alpha=0.7,
             label="Legit", color="#2ecc71")
axes[0,1].hist(df[df.Class==1]["Amount"], bins=50, alpha=0.7,
             label="Fraud", color="#e74c3c")
axes[0,1].set_title("Transaction Amount")
axes[0,1].legend()

# 3. Hour distribution
axes[0,2].hist(df[df.Class==0]["Hour"], bins=24, alpha=0.7,
             label="Legit", color="#2ecc71")
axes[0,2].hist(df[df.Class==1]["Hour"], bins=24, alpha=0.7,
             label="Fraud", color="#e74c3c")
axes[0,2].set_title("Transaction Hour")
axes[0,2].legend()

# 4. Merchant distance boxplot
axes[1,0].boxplot([df[df.Class==0]["MerchantDistance"],
                   df[df.Class==1]["MerchantDistance"]],
                  labels=["Legit", "Fraud"])
axes[1,0].set_title("Merchant Distance (km)")

# 5. Correlation heatmap
corr = df.corr()[["Class"]].drop("Class").sort_values("Class")
sns.heatmap(corr, ax=axes[1,1], annot=True, fmt=".2f", cmap="RdYlGn")
axes[1,1].set_title("Correlation with Fraud")

# 6. Transaction frequency
axes[1,2].hist(df[df.Class==0]["TransactionFreq"], bins=20,
             alpha=0.7, label="Legit", color="#2ecc71")
axes[1,2].hist(df[df.Class==1]["TransactionFreq"], bins=20,
             alpha=0.7, label="Fraud", color="#e74c3c")
axes[1,2].set_title("Transaction Frequency")
axes[1,2].legend()

plt.tight_layout()
plt.savefig("outputs/eda_dashboard.png", dpi=150)
plt.show()
print("✅ EDA charts saved to outputs/eda_dashboard.png")