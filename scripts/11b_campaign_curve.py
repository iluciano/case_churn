import pandas as pd
import numpy as np

s = pd.read_parquet("data/processed/scores_test.parquet")
y = s["churn_3m"].astype(int)
p = s["p_churn"].to_numpy()

def recall_at_k(frac: float):
    k = max(1, int(len(y) * frac))
    idx = np.argsort(-p)[:k]
    return (y.iloc[idx].sum() / y.sum()) if y.sum() else 0.0, k

def churnrate_at_k(frac: float):
    k = max(1, int(len(y) * frac))
    idx = np.argsort(-p)[:k]
    return float(y.iloc[idx].mean()), k

for f in [0.01, 0.03, 0.05, 0.10, 0.20]:
    r, k = recall_at_k(f)
    cr, _ = churnrate_at_k(f)
    print(f"Top {int(f*100):>2}% | k={k:>7} | Recall={r:.3f} | ChurnRate={cr:.3f}")