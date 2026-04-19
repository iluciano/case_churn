"""
11b_campaign_curve.py  –  Curva de campanha Top-K
==================================================
CORREÇÃO v2: substituído path relativo hardcoded por PROJECT_ROOT.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCORES_PATH  = PROJECT_ROOT / "data" / "processed" / "scores_test.parquet"

if not SCORES_PATH.exists():
    raise FileNotFoundError(
        f"Não encontrei {SCORES_PATH}.\n"
        "Execute antes: 11_train_churn_model.py"
    )

s = pd.read_parquet(SCORES_PATH)
y = s["churn_3m"].astype(int)
p = s["p_churn"].to_numpy()

base_churn = float(y.mean())
print(f"Base churn TEST : {base_churn:.4f}  ({base_churn*100:.2f}%)")
print(f"Total clientes  : {len(y):,}")
print(f"Total churners  : {int(y.sum()):,}")
print()


def recall_at_k(frac: float):
    k   = max(1, int(len(y) * frac))
    idx = np.argsort(-p)[:k]
    return (y.iloc[idx].sum() / y.sum()) if y.sum() else 0.0, k


def churnrate_at_k(frac: float):
    k   = max(1, int(len(y) * frac))
    idx = np.argsort(-p)[:k]
    return float(y.iloc[idx].mean()), k


header = f"{'Top %':>6} | {'k':>8} | {'Recall':>8} | {'ChurnRate':>10} | {'Lift':>6}"
print(header)
print("-" * len(header))

for f in [0.01, 0.03, 0.05, 0.10, 0.20]:
    r,  k  = recall_at_k(f)
    cr, _  = churnrate_at_k(f)
    lift   = cr / base_churn if base_churn > 0 else float("nan")
    print(f"{int(f*100):>5}% | {k:>8,} | {r:>8.3f} | {cr:>10.3f} | {lift:>6.2f}x")
