import numpy as np
import pandas as pd
import os

os.makedirs("outputs", exist_ok=True)

# Set seed for reproducibility
np.random.seed(42)

N = 10000          # total transactions
n_fraud = 500      # 5% fraud
n_legit = N - n_fraud

def generate(n, is_fraud):
    if is_fraud:
        # Fraud: high amounts, late night, far merchant
        amount   = np.random.exponential(400, n).clip(10, 5000)
        hour     = np.random.choice([*range(0,6), *range(22,24)], n)
        v1       = np.random.normal(-3, 2, n)
        v2       = np.random.normal(4, 2, n)
        v3       = np.random.normal(-2, 1.5, n)
        v4       = np.random.normal(3, 1.5, n)
        v5       = np.random.normal(-1.5, 1, n)
        freq     = np.random.randint(5, 20, n)
        dist     = np.random.uniform(500, 5000, n)
        age      = np.random.randint(18, 45, n)
        label    = np.ones(n, dtype=int)
    else:
        # Legit: normal amounts, daytime, nearby merchant
        amount   = np.random.exponential(80, n).clip(1, 2000)
        hour     = np.random.randint(6, 22, n)
        v1       = np.random.normal(0.5, 1, n)
        v2       = np.random.normal(-0.5, 1, n)
        v3       = np.random.normal(0.3, 1, n)
        v4       = np.random.normal(-0.3, 1, n)
        v5       = np.random.normal(0.1, 1, n)
        freq     = np.random.randint(1, 5, n)
        dist     = np.random.uniform(0, 100, n)
        age      = np.random.randint(25, 75, n)
        label    = np.zeros(n, dtype=int)

    return pd.DataFrame({
        "Amount": amount, "Hour": hour,
        "V1": v1, "V2": v2, "V3": v3, "V4": v4, "V5": v5,
        "TransactionFreq": freq,
        "MerchantDistance": dist,
        "CardholderAge": age,
        "Class": label
    })

# Combine & shuffle
df = pd.concat([
    generate(n_legit, False),
    generate(n_fraud, True)
], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv("outputs/dataset.csv", index=False)
print(f"✅ Dataset saved: {len(df):,} rows | Fraud: {df['Class'].sum()} ({df['Class'].mean()*100:.1f}%)")