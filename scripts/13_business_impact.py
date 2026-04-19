"""
13_business_impact.py  –  Avaliação econômica e análise de sensibilidade
=========================================================================
CORREÇÕES v2:
  1. MOEDA CORRIGIDA: valores em NTD (New Taiwan Dollar), conforme dicionário
     de dados. A versão original usava R$ incorretamente.

  2. ARPU calculado apenas nos meses do conjunto de TESTE, não em toda a
     model_table (treino + validação + teste). Usar o ARPU dos meses de
     avaliação é mais coerente com a análise financeira que se segue.

  3. Adicionada nota metodológica sobre premissas do modelo econômico.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_TABLE  = PROJECT_ROOT / "data" / "processed" / "model_table"
SCORES_TEST  = PROJECT_ROOT / "data" / "processed" / "scores_test.parquet"
OUT_DIR      = PROJECT_ROOT / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "churn_3m"

# Parâmetros de sensibilidade
TOP_FRACS        = [0.01, 0.03, 0.05, 0.10, 0.20]
FREE_MONTHS_LIST = [1, 2, 3]
ACCEPT_RATE_LIST = [0.3, 0.5, 0.7]
RETENTION_MONTHS = 12   # hipótese do case: quem aceita fica 1 ano


def compute_arpu(model_df: pd.DataFrame, test_safras: list) -> float:
    """
    CORREÇÃO: ARPU calculado apenas nos meses do conjunto de TESTE,
    para manter consistência com a análise econômica.

    A moeda é NTD (New Taiwan Dollar), conforme o dicionário de dados:
      'plan_list_price: in New Taiwan Dollar (NTD)'
    """
    if "paid_sum" not in model_df.columns:
        raise ValueError("model_table não tem 'paid_sum'.")

    test_mask = model_df["safra"].isin(test_safras) if test_safras else slice(None)
    paid      = pd.to_numeric(model_df.loc[test_mask, "paid_sum"], errors="coerce").fillna(0.0)
    paid_pos  = paid[paid > 0]

    if len(paid_pos) == 0:
        raise ValueError("Não encontrei paid_sum>0 para calcular ARPU.")

    arpu = float(paid_pos.mean())
    print(f"ARPU (NTD, meses de teste) = {arpu:.4f}")
    print(f"  n_clientes com paid_sum>0: {len(paid_pos):,}")
    print(f"  Nota: valores em NTD (New Taiwan Dollar), conforme dicionário de dados.")
    return arpu


def compute_counts_for_topk(scores: pd.DataFrame, frac: float):
    y   = scores[TARGET].astype(int)
    p   = scores["p_churn"].to_numpy()
    k   = max(1, int(len(scores) * frac))
    idx = np.argsort(-p)[:k]
    sel = scores.iloc[idx]
    return k, int((sel[TARGET] == 1).sum()), int((sel[TARGET] == 0).sum())


def expected_net(arpu: float, tp: int, fp: int,
                 free_months: int, accept_rate: float) -> tuple[float, float, float]:
    """
    Modelo econômico simplificado (hipóteses do case):
    - TP que aceita: permanece (12 - free_months) meses pagando.
      Ganho = accept_rate × TP × (12 - free_months) × ARPU
    - FP acionado:  recebe meses grátis desnecessariamente.
      Custo = FP × free_months × ARPU

    Limitações assumidas:
    - Não modela elasticidade da oferta além da taxa de aceitação
    - ARPU fixo (não considera upgrades pós-retenção)
    - Taxa de aceitação uniforme entre clusters (no modelo real, variaria)
    """
    paid_months = max(0, RETENTION_MONTHS - free_months)
    gain = accept_rate * tp * paid_months * arpu
    cost = fp * free_months * arpu
    net  = gain - cost
    return gain, cost, net


def main():
    if not MODEL_TABLE.exists():
        raise FileNotFoundError(f"Não encontrei {MODEL_TABLE}. Rode o script 10.")
    if not SCORES_TEST.exists():
        raise FileNotFoundError(f"Não encontrei {SCORES_TEST}. Rode o script 11.")

    model  = pd.read_parquet(MODEL_TABLE)
    scores = pd.read_parquet(SCORES_TEST)

    if "p_churn" not in scores.columns or TARGET not in scores.columns:
        raise ValueError("scores_test.parquet precisa ter p_churn e churn_3m.")

    # Identifica safras do conjunto de teste para ARPU consistente
    test_safras = sorted(scores["safra"].dropna().unique().tolist())
    print(f"Safras do conjunto de teste: {test_safras}")

    # CORREÇÃO: ARPU apenas do período de teste + moeda NTD
    arpu = compute_arpu(model, test_safras)

    base_rate = float(scores[TARGET].mean())
    print(f"\nBase churn rate (TEST): {base_rate:.4f}  ({base_rate*100:.2f}%)")

    rows = []
    for free_m in FREE_MONTHS_LIST:
        for acc in ACCEPT_RATE_LIST:
            for frac in TOP_FRACS:
                k, tp, fp   = compute_counts_for_topk(scores, frac)
                gain, cost, net = expected_net(arpu, tp, fp, free_m, acc)
                precision_k = tp / k if k else 0.0
                rows.append({
                    "free_months":          free_m,
                    "accept_rate":          acc,
                    "top_frac":             frac,
                    "k":                    k,
                    "TP":                   tp,
                    "FP":                   fp,
                    "precision_at_k":       precision_k,
                    "base_churn_rate_test": base_rate,
                    "lift":                 precision_k / base_rate if base_rate > 0 else np.nan,
                    "ARPU_NTD":             arpu,
                    "expected_gain_NTD":    gain,
                    "expected_cost_NTD":    cost,
                    "expected_net_NTD":     net,
                })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "business_impact_sensitivity.csv", index=False)
    print(f"\nSensibilidade → {OUT_DIR / 'business_impact_sensitivity.csv'}")

    best = (
        out.sort_values("expected_net_NTD", ascending=False)
           .groupby(["free_months", "accept_rate"], as_index=False)
           .head(1)
           .reset_index(drop=True)
    )
    best.to_csv(OUT_DIR / "best_policy_by_accept_free.csv", index=False)

    print("\n=== Melhor política por (free_months, accept_rate) ===")
    cols_show = ["free_months", "accept_rate", "top_frac", "k",
                 "TP", "FP", "precision_at_k", "lift", "expected_net_NTD"]
    print(best[cols_show].to_string(index=False))

    print("\nNOTA SOBRE MOEDA:")
    print("  Todos os valores financeiros estão em NTD (New Taiwan Dollar).")
    print("  Referência: dicionário de dados — 'plan_list_price: in New Taiwan Dollar'.")
    print("  Para converter a BRL: usar câmbio de referência (ex: 1 NTD ≈ 0.17 BRL em 2016).")
    print("\nNOTA METODOLÓGICA:")
    print("  Trate os resultados como análise de sensibilidade, não como ROI garantido.")
    print("  A conclusão depende fortemente de aceitação da oferta, ARPU e fração Top-K.")


if __name__ == "__main__":
    main()
