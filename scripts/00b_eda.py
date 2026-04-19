"""
00b_eda.py  –  Análise Exploratória de Dados
==============================================
Objetivo: levantar hipóteses voltadas ao problema de churn e orientar
decisões de feature engineering e modelagem.

Saídas (reports/eda/):
  - eda_churn_by_month.csv
  - eda_missing_summary.csv
  - eda_numeric_stats.csv
  - eda_autorenew_vs_churn.csv
  - eda_figures/  (PNGs)

Critério do case atendido:
  "Analisou hipóteses voltadas ao problema e para a modelagem?"
  "Compreendeu as implicações das análises?"
  "Utilizou o resultado das análises para orientar decisões?"
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # sem display, compatível com VS Code headless
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ─────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_TABLE  = PROJECT_ROOT / "data" / "processed" / "model_table"
OUT_DIR      = PROJECT_ROOT / "reports" / "eda"
FIG_DIR      = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "churn_3m"

# ─────────────────────────────────────────────
def safe_qcut(series: pd.Series, q: int = 5, fillna_value: float | int | None = None) -> pd.Series:
    """
    Versão robusta de qcut para cenários com muitos valores repetidos.
    Evita erro quando duplicates="drop" reduz a quantidade de bins.
    """
    s = pd.to_numeric(series, errors="coerce")
    if fillna_value is not None:
        s = s.fillna(fillna_value)

    nunique = s.nunique(dropna=True)
    if nunique == 0:
        return pd.Series(["Faixa única"] * len(s), index=s.index, dtype="object")

    q_eff = min(q, nunique)
    probe = pd.qcut(s, q=q_eff, duplicates="drop")
    n_bins = len(probe.cat.categories)

    if n_bins <= 1:
        return pd.Series(["Faixa única"] * len(s), index=s.index, dtype="object")

    labels = [f"Q{i+1}" for i in range(n_bins)]
    labels[0] = "Q1(menor)"
    labels[-1] = f"Q{n_bins}(maior)"

    return pd.qcut(s, q=q_eff, labels=labels, duplicates="drop")

# ─────────────────────────────────────────────
def save(fig: plt.Figure, name: str) -> None:
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  salvo: {path.name}")


def load_trusted(model_table: Path) -> pd.DataFrame:
    df = pd.read_parquet(model_table)
    df["safra"]       = pd.to_numeric(df["safra"],       errors="coerce").astype("Int32")
    df[TARGET]        = pd.to_numeric(df[TARGET],         errors="coerce").fillna(0).astype("int8")
    df["is_ativo"]    = pd.to_numeric(df["is_ativo"],     errors="coerce").fillna(0).astype("int8")
    df["label_trust"] = pd.to_numeric(df["label_trust"],  errors="coerce").fillna(0).astype("int8")

    # apenas elegíveis e meses confiáveis
    df = df[(df["is_ativo"] == 1) & (df["label_trust"] == 1)].copy()
    print(f"Linhas após filtro (ativo + label_trust): {len(df):,}")
    print(f"Meses: {int(df['safra'].min())} → {int(df['safra'].max())} | n_months: {df['safra'].nunique()}")
    print(f"Churn rate geral: {df[TARGET].mean():.4f}")
    return df


# ─────────────────────────────────────────────
# 1. Churn rate por safra
# ─────────────────────────────────────────────
def eda_churn_by_month(df: pd.DataFrame) -> None:
    """
    Hipótese: a taxa de churn varia ao longo do tempo (sazonalidade, coortes).
    Implicação: split out-of-time é mandatório; embaralhar meses vazaria informação temporal.
    """
    g = df.groupby("safra").agg(
        rows    =(TARGET, "count"),
        churn   =(TARGET, "mean"),
        churn_n =(TARGET, "sum"),
    ).reset_index()
    g.to_csv(OUT_DIR / "eda_churn_by_month.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(10, 4.5))
    ax2 = ax1.twinx()

    safras = g["safra"].astype(str)
    ax1.bar(safras, g["rows"] / 1e3, color="steelblue", alpha=0.4, label="Volume (mil)")
    ax2.plot(safras, g["churn"] * 100, marker="o", color="crimson", linewidth=2, label="Churn (%)")
    ax2.axhline(df[TARGET].mean() * 100, linestyle="--", color="gray", label="Média geral")

    ax1.set_ylabel("Volume (mil clientes)")
    ax2.set_ylabel("Churn (%)")
    ax1.set_xlabel("Safra")
    ax1.tick_params(axis="x", rotation=45)
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter())

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    ax1.set_title("Churn rate e volume por safra (meses confiáveis)", fontsize=13)
    fig.tight_layout()
    save(fig, "01_churn_by_month.png")
    print(g.to_string(index=False))


# ─────────────────────────────────────────────
# 2. Missings por coluna
# ─────────────────────────────────────────────
def eda_missings(df: pd.DataFrame) -> None:
    """
    Implicação para modelagem: colunas com >50% missing devem ser avaliadas
    para remoção ou imputação especial.
    """
    miss = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    miss = miss[miss > 0].reset_index()
    miss.columns = ["feature", "pct_missing"]
    miss.to_csv(OUT_DIR / "eda_missing_summary.csv", index=False)

    top = miss.head(20)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top["feature"][::-1], top["pct_missing"][::-1], color="coral")
    ax.axvline(50, linestyle="--", color="red", label="50% threshold")
    ax.set_xlabel("% missing")
    ax.set_title("Top 20 features com missing (%)", fontsize=13)
    ax.legend()
    fig.tight_layout()
    save(fig, "02_missings.png")

    print("\nTop 10 missings:")
    print(miss.head(10).to_string(index=False))


# ─────────────────────────────────────────────
# 3. Estatísticas descritivas das features numéricas
# ─────────────────────────────────────────────
def eda_numeric_stats(df: pd.DataFrame) -> None:
    num_cols = [c for c in df.select_dtypes("number").columns
                if c not in {TARGET, "safra", "is_ativo", "label_trust",
                             "is_ativo_fut", "fut_active_rate", "churn_rate_month"}]
    stats = df[num_cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    stats.to_csv(OUT_DIR / "eda_numeric_stats.csv")
    print("\nStatísticas numéricas salvas.")


# ─────────────────────────────────────────────
# 4. Hipótese: auto_renew_rate baixa → maior churn
# ─────────────────────────────────────────────
def eda_autorenew_vs_churn(df: pd.DataFrame) -> None:
    """
    Hipótese de negócio central: clientes sem renovação automática têm
    maior probabilidade de churn.
    """
    if "auto_renew_rate" not in df.columns:
        print("Coluna auto_renew_rate não encontrada, pulando.")
        return

    # Discretiza em quintis de forma robusta
    df = df.copy()
    df["autorenew_bin"] = safe_qcut(df["auto_renew_rate"], q=5, fillna_value=-1)
    g = df.groupby("autorenew_bin")[TARGET].mean().reset_index()
    g.columns = ["autorenew_bin", "churn_rate"]
    g.to_csv(OUT_DIR / "eda_autorenew_vs_churn.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(g["autorenew_bin"].astype(str), g["churn_rate"] * 100, color="steelblue")
    ax.axhline(df[TARGET].mean() * 100, linestyle="--", color="gray", label="Média")
    ax.set_xlabel("Quintil de auto_renew_rate")
    ax.set_ylabel("Churn (%)")
    ax.set_title("Hipótese: auto_renew_rate baixa → maior churn", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend()
    fig.tight_layout()
    save(fig, "03_autorenew_vs_churn.png")


# ─────────────────────────────────────────────
# 5. Hipótese: engajamento baixo (total_secs) → maior churn
# ─────────────────────────────────────────────
def eda_engagement_vs_churn(df: pd.DataFrame) -> None:
    """
    Hipótese: clientes que ouvem menos têm maior propensão a cancelar.
    Implicação: total_secs e num_unq devem ser features relevantes.
    """
    if "total_secs" not in df.columns:
        print("Coluna total_secs não encontrada, pulando.")
        return

    df = df.copy()
    df["secs_bin"] = safe_qcut(df["total_secs"].clip(lower=0) + 1, q=5, fillna_value=0)
    g = df.groupby("secs_bin")[TARGET].mean().reset_index()
    g.columns = ["secs_bin", "churn_rate"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(g["secs_bin"].astype(str), g["churn_rate"] * 100, color="teal")
    ax.axhline(df[TARGET].mean() * 100, linestyle="--", color="gray", label="Média")
    ax.set_xlabel("Quintil de total_secs")
    ax.set_ylabel("Churn (%)")
    ax.set_title("Hipótese: engajamento baixo → maior churn", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend()
    fig.tight_layout()
    save(fig, "04_engagement_vs_churn.png")


# ─────────────────────────────────────────────
# 6. Hipótese: queda no pagamento (paid_sum_diff1) → sinal de churn iminente
# ─────────────────────────────────────────────
def eda_paid_diff_vs_churn(df: pd.DataFrame) -> None:
    """
    Hipótese: variação negativa no valor pago (paid_sum_diff1 < 0) indica
    downgrade ou não-renovação em andamento.
    Implicação: features de tendência (diff1) devem ser priorizadas.
    """
    if "paid_sum_diff1" not in df.columns:
        print("Coluna paid_sum_diff1 não encontrada. Rode 01b_add_lag_features.py antes.")
        return

    df = df.copy()
    df["paid_trend"] = pd.cut(
        df["paid_sum_diff1"].fillna(0),
        bins=[-np.inf, -1, 1, np.inf],
        labels=["Queda", "Estável", "Alta"]
    )
    g = df.groupby("paid_trend")[TARGET].agg(["mean", "count"]).reset_index()
    g.columns = ["paid_trend", "churn_rate", "n"]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {"Queda": "crimson", "Estável": "steelblue", "Alta": "seagreen"}
    ax.bar(g["paid_trend"].astype(str),
           g["churn_rate"] * 100,
           color=[colors[t] for t in g["paid_trend"].astype(str)])
    ax.axhline(df[TARGET].mean() * 100, linestyle="--", color="gray", label="Média")
    ax.set_xlabel("Variação mensal no pagamento (paid_sum_diff1)")
    ax.set_ylabel("Churn (%)")
    ax.set_title("Hipótese: queda no pagamento → churn iminente", fontsize=12)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend()
    fig.tight_layout()
    save(fig, "05_paid_diff_vs_churn.png")
    print("\nChurn por tendência de pagamento:")
    print(g.to_string(index=False))


# ─────────────────────────────────────────────
# 7. Distribuição de idade (bd) — tratamento de outliers
# ─────────────────────────────────────────────
def eda_age_distribution(df: pd.DataFrame) -> None:
    """
    O dicionário de dados avisa: bd tem outliers de -7000 a 2015.
    Mostra antes/depois do tratamento aplicado no script 01_build_features.py
    (mantém apenas 10 ≤ bd ≤ 90).
    """
    if "bd" not in df.columns:
        print("Coluna bd não encontrada, pulando.")
        return

    bd_valid = df["bd"].dropna()
    bd_valid = bd_valid[(bd_valid >= 10) & (bd_valid <= 90)]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(bd_valid, bins=40, color="steelblue", edgecolor="white")
    ax.set_xlabel("Idade (bd) — após remoção de outliers")
    ax.set_ylabel("Frequência")
    ax.set_title(f"Distribuição de idade\n"
                 f"(válidos: {len(bd_valid):,} | missing/outlier: {df['bd'].isna().sum():,})", fontsize=12)
    fig.tight_layout()
    save(fig, "06_age_distribution.png")


# ─────────────────────────────────────────────
# 8. Correlação entre features numéricas principais e target
# ─────────────────────────────────────────────
def eda_correlations(df: pd.DataFrame) -> None:
    """
    Visão rápida de quais features têm correlação linear com o churn.
    Serve como ponto de partida para feature selection.
    """
    focus = [
        "total_secs", "num_unq", "plays_total", "paid_sum",
        "auto_renew_rate", "cancel_txn_count", "has_usage", "has_payment",
        "total_secs_diff1", "paid_sum_diff1", "has_usage_diff1",
    ]
    avail = [c for c in focus if c in df.columns] + [TARGET]
    corr  = df[avail].corr()[[TARGET]].drop(TARGET).sort_values(TARGET)

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["crimson" if v > 0 else "steelblue" for v in corr[TARGET]]
    ax.barh(corr.index, corr[TARGET], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Correlação de Pearson com {TARGET}")
    ax.set_title("Correlação linear: features principais × churn", fontsize=12)
    fig.tight_layout()
    save(fig, "07_correlations_with_churn.png")

    print("\nCorrelações com churn_3m:")
    print(corr.to_string())


# ─────────────────────────────────────────────
def main():
    print("=" * 60)
    print("EDA — Análise Exploratória")
    print("=" * 60)

    if not MODEL_TABLE.exists():
        raise FileNotFoundError(
            f"Não encontrei {MODEL_TABLE}.\n"
            "Execute antes: 01_build_features.py → 01b_add_lag_features.py → 10_build_model_table.py"
        )

    df = load_trusted(MODEL_TABLE)

    print("\n[1/7] Churn por safra...")
    eda_churn_by_month(df)

    print("\n[2/7] Missings...")
    eda_missings(df)

    print("\n[3/7] Estatísticas descritivas...")
    eda_numeric_stats(df)

    print("\n[4/7] Auto-renew vs churn...")
    eda_autorenew_vs_churn(df)

    print("\n[5/7] Engajamento vs churn...")
    eda_engagement_vs_churn(df)

    print("\n[6/7] Variação de pagamento vs churn...")
    eda_paid_diff_vs_churn(df)

    print("\n[7/7] Distribuição de idade...")
    eda_age_distribution(df)
    eda_correlations(df)

    print(f"\nOK → todos os arquivos em: {OUT_DIR.resolve()}")
    print(f"     figuras em:            {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()
