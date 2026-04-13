from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_TABLE = PROJECT_ROOT / "data" / "processed" / "model_table"          # pasta parquet
SCORES_TEST = PROJECT_ROOT / "data" / "processed" / "scores_test.parquet"  # arquivo
OUT_DIR = PROJECT_ROOT / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "churn_3m"

TOP_FRACS = [0.01, 0.03, 0.05, 0.10, 0.20]
FREE_MONTHS_LIST = [1, 2, 3]               # sensibilidade: 1,2,3 meses grátis
ACCEPT_RATE_LIST = [0.3, 0.5, 0.7]         # sensibilidade: 30%, 50%, 70% aceitam e ficam 1 ano

# Hipótese do case: quem aceita fica 1 ano; meses grátis são dentro desse ano
RETENTION_MONTHS = 12


def compute_arpu(model_df: pd.DataFrame) -> float:
    """
    ARPU mensal proxy:
    - usa paid_sum (valor pago no mês)
    - média apenas onde paid_sum>0
    """
    if "paid_sum" not in model_df.columns:
        raise ValueError("model_table não tem 'paid_sum'. Verifique o script 01/10.")
    paid = pd.to_numeric(model_df["paid_sum"], errors="coerce").fillna(0.0)
    paid_pos = paid[paid > 0]
    if len(paid_pos) == 0:
        raise ValueError("Não encontrei paid_sum>0 para calcular ARPU.")
    return float(paid_pos.mean())


def compute_counts_for_topk(scores: pd.DataFrame, frac: float):
    """
    Retorna TP/FP no público acionado Top frac por p_churn.
    """
    y = scores[TARGET].astype(int)
    p = scores["p_churn"].to_numpy()

    k = max(1, int(len(scores) * frac))
    idx = np.argsort(-p)[:k]
    sel = scores.iloc[idx]

    tp = int((sel[TARGET] == 1).sum())
    fp = int((sel[TARGET] == 0).sum())
    return k, tp, fp


def expected_net(arpu: float, tp: int, fp: int, free_months: int, accept_rate: float) -> tuple[float, float, float]:
    """
    Modelo do case:
    - Se é TP (ia churnar) e acionamos:
        accept_rate continuam 1 ano.
        Dentro do ano, free_months são grátis => pagam (12-free_months) meses.
        Ganho = accept_rate * TP * (12-free_months) * ARPU
    - Se é FP (não ia churnar) e acionamos:
        custo = FP * free_months * ARPU (meses grátis "desnecessários")
    """
    paid_months = max(0, RETENTION_MONTHS - free_months)
    gain = accept_rate * tp * paid_months * arpu
    cost = fp * free_months * arpu
    net = gain - cost
    return gain, cost, net


def main():
    if not MODEL_TABLE.exists():
        raise FileNotFoundError(f"Não encontrei {MODEL_TABLE}. Rode o script 10.")
    if not SCORES_TEST.exists():
        raise FileNotFoundError(f"Não encontrei {SCORES_TEST}. Rode o script 11.")

    model = pd.read_parquet(MODEL_TABLE)
    scores = pd.read_parquet(SCORES_TEST)

    if "p_churn" not in scores.columns or TARGET not in scores.columns:
        raise ValueError("scores_test.parquet precisa ter colunas p_churn e churn_3m.")

    arpu = compute_arpu(model)
    print(f"ARPU (auto) = {arpu:.4f}")

    rows = []
    for free_m in FREE_MONTHS_LIST:
        for acc in ACCEPT_RATE_LIST:
            for frac in TOP_FRACS:
                k, tp, fp = compute_counts_for_topk(scores, frac)
                gain, cost, net = expected_net(arpu, tp, fp, free_m, acc)

                precision_k = tp / k if k else 0.0
                base_rate = float(scores[TARGET].mean())

                rows.append({
                    "free_months": free_m,
                    "accept_rate": acc,
                    "top_frac": frac,
                    "k": k,
                    "TP": tp,
                    "FP": fp,
                    "precision_at_k": precision_k,      # churn rate no público acionado
                    "base_churn_rate_test": base_rate,
                    "lift": (precision_k / base_rate) if base_rate > 0 else np.nan,
                    "ARPU_used": arpu,
                    "expected_gain": gain,
                    "expected_cost": cost,
                    "expected_net": net,
                })

    out = pd.DataFrame(rows)

    out_path = OUT_DIR / "business_impact_sensitivity.csv"
    out.to_csv(out_path, index=False)
    print("OK ->", out_path)

    # Melhor política (maior net) por cenário (free_months, accept_rate)
    best = (
        out.sort_values("expected_net", ascending=False)
           .groupby(["free_months", "accept_rate"], as_index=False)
           .head(1)
           .reset_index(drop=True)
    )

    best_path = OUT_DIR / "best_policy_by_accept_free.csv"
    best.to_csv(best_path, index=False)
    print("OK ->", best_path)

    print("\n=== Best policy per (free_months, accept_rate) ===")
    cols_show = ["free_months","accept_rate","top_frac","k","TP","FP","precision_at_k","lift","expected_net"]
    print(best[cols_show].to_string(index=False))


if __name__ == "__main__":
    main()