from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT    = Path(__file__).resolve().parents[1]
MODEL_TABLE_DIR = PROJECT_ROOT / "data" / "processed" / "model_table"
SCORES_TEST     = PROJECT_ROOT / "data" / "processed" / "scores_test.parquet"
OUT_DIR         = PROJECT_ROOT / "reports"
FIG_DIR         = OUT_DIR / "figures_unsup"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "churn_3m"
MAX_ROWS_CLUSTERING = 300_000
CHOOSE_K_SAMPLE     = 30_000
SILHOUETTE_SAMPLE   = 30_000


def save_fig(fig: plt.Figure, name: str) -> None:
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  figura → {path.name}")


def reduce_memory(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_integer_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast="integer")
        elif pd.api.types.is_float_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast="float")
    return df


def deduplicate_keys(df: pd.DataFrame, keys: list[str], name: str) -> pd.DataFrame:
    dup = int(df.duplicated(keys).sum())
    if dup:
        print(f"[AVISO] {name} tem {dup:,} chaves duplicadas em {keys}. Mantendo a última ocorrência.")
        df = df.drop_duplicates(keys, keep="last").copy()
    return df


def stratified_sample_by_month(df: pd.DataFrame, max_rows: int, month_col: str = "safra",
                               random_state: int = 42) -> pd.DataFrame:
    if len(df) <= max_rows:
        print(f"Amostra não necessária. Linhas para clustering: {len(df):,}")
        return df.copy()

    frac = max_rows / len(df)
    pieces = []
    for _, g in df.groupby(month_col, dropna=False):
        n = max(1, int(round(len(g) * frac)))
        n = min(n, len(g))
        pieces.append(g.sample(n=n, random_state=random_state))
    out = pd.concat(pieces, ignore_index=True)

    if len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=random_state).reset_index(drop=True)

    print(f"Amostra estratificada por safra: {len(df):,} → {len(out):,} linhas")
    return out


def pick_numeric_features(df: pd.DataFrame) -> list[str]:
    drop = {
        "safra", "safra_fut", "msno", TARGET,
        "p_churn", "is_ativo", "is_ativo_fut",
        "pred_action", "fp", "fn",
        "label_trust", "fut_active_rate", "churn_rate_month",
    }
    num_cols = [c for c in df.columns
                if pd.api.types.is_numeric_dtype(df[c]) and c not in drop]

    preferred = [
        "num_unq", "total_secs", "plays_total",
        "paid_sum", "auto_renew_rate", "cancel_txn_count",
        "has_usage", "has_payment",
        "total_secs_lag1",  "total_secs_diff1",
        "num_unq_lag1",     "num_unq_diff1",
        "paid_sum_lag1",    "paid_sum_diff1",
        "plays_total_lag1", "plays_total_diff1",
        "has_usage_lag1",   "has_usage_diff1",
        "has_payment_lag1", "has_payment_diff1",
    ]
    cols = [c for c in preferred if c in num_cols] or num_cols
    return cols[:40]


def cluster_profile(df: pd.DataFrame, label_col: str = "cluster") -> pd.DataFrame:
    agg = {"msno": "count", "p_churn": "mean", TARGET: "mean", "fp": "mean", "fn": "mean"}
    for c in ["paid_sum", "total_secs", "num_unq", "auto_renew_rate", "cancel_txn_count"]:
        if c in df.columns:
            agg[c] = "mean"
    out = df.groupby(label_col).agg(agg).rename(columns={"msno": "rows"}).reset_index()
    out["share"] = out["rows"] / out["rows"].sum()
    return out.sort_values("p_churn", ascending=False)


def simple_drivers(df, feature_cols, label_col="cluster", top_n=15):
    eps = 1e-9
    overall_var = df[feature_cols].var(numeric_only=True)
    rows = []
    for c in feature_cols:
        means = df.groupby(label_col)[c].mean()
        score = float(means.var() / (overall_var.get(c, eps) + eps))
        rows.append({"feature": c, "separation_score": score})
    return pd.DataFrame(rows).sort_values("separation_score", ascending=False).head(top_n)


def monthly_drift(df, label_col="cluster"):
    g = df.groupby(["safra", label_col]).agg(
        rows         =("msno",    "count"),
        p_churn_mean =("p_churn", "mean"),
        churn_rate   =(TARGET,    "mean"),
    ).reset_index()
    total = g.groupby("safra")["rows"].transform("sum")
    g["share"] = g["rows"] / total
    return g.sort_values(["safra", label_col])


def choose_k(X_scaled: np.ndarray, k_range=range(2, 11), sample_size: int = CHOOSE_K_SAMPLE) -> int:
    np.random.seed(42)
    idx = np.random.choice(len(X_scaled), min(sample_size, len(X_scaled)), replace=False)
    Xs  = X_scaled[idx]

    inertias = []
    sil_vals = []

    print(f"\n  Avaliando k de {k_range.start} a {k_range.stop - 1} "
          f"(amostra={len(Xs):,} pontos)...")

    for k in k_range:
        km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=4096, n_init=3)
        labels = km.fit_predict(Xs)
        inertias.append(float(km.inertia_))
        sil = silhouette_score(Xs, labels, sample_size=min(10_000, len(Xs)), random_state=42)
        sil_vals.append(float(sil))
        print(f"    k={k:>2} | inertia={km.inertia_:,.0f} | silhouette={sil:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(list(k_range), inertias, marker="o")
    axes[0].set_xlabel("Número de clusters (k)")
    axes[0].set_ylabel("Inertia")
    axes[0].set_title("Elbow Method")
    axes[0].grid(alpha=0.25)

    best_k_idx = int(np.argmax(sil_vals))
    best_k     = list(k_range)[best_k_idx]
    axes[1].plot(list(k_range), sil_vals, marker="o")
    axes[1].axvline(best_k, linestyle="--", label=f"Melhor k={best_k} (sil={sil_vals[best_k_idx]:.3f})")
    axes[1].set_xlabel("Número de clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Score por k")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.suptitle("Escolha de k — MiniBatchKMeans", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "00_elbow_silhouette.png")

    print(f"\n  → k escolhido: {best_k} (maior silhouette={sil_vals[best_k_idx]:.4f})")
    print("  Interpretação: escolha orientada por critério exploratório, não causal.")
    return best_k


def main():
    print("PROJECT_ROOT:", PROJECT_ROOT)

    if not MODEL_TABLE_DIR.exists():
        raise FileNotFoundError(f"Não encontrei {MODEL_TABLE_DIR}.")
    if not SCORES_TEST.exists():
        raise FileNotFoundError(f"Não encontrei {SCORES_TEST}.")

    print("\n[0/5] Carregando dados...")
    model  = pd.read_parquet(MODEL_TABLE_DIR)
    scores = pd.read_parquet(SCORES_TEST)

    print(f"model shape : {model.shape}")
    print(f"scores shape: {scores.shape}")

    model  = model.drop(columns=[TARGET], errors="ignore")

    model  = reduce_memory(model)
    scores = reduce_memory(scores)

    key_cols = ["msno", "safra"]
    model  = deduplicate_keys(model,  key_cols, "model")
    scores = deduplicate_keys(scores, key_cols, "scores")

    print(f"Meses em scores: {int(pd.to_numeric(scores['safra'], errors='coerce').min())} "
          f"→ {int(pd.to_numeric(scores['safra'], errors='coerce').max())} | "
          f"n={pd.to_numeric(scores['safra'], errors='coerce').nunique()}")

    score_months = pd.to_numeric(scores["safra"], errors="coerce").dropna().unique().tolist()
    if "safra" in model.columns:
        model["safra"] = pd.to_numeric(model["safra"], errors="coerce").astype("Int32")
        before = len(model)
        model = model[model["safra"].isin(score_months)].copy()
        print(f"Filtro de meses do model: {before:,} → {len(model):,}")

    df = model.merge(scores[["msno", "safra", "p_churn", TARGET]],
                     on=["msno", "safra"], how="inner", validate="one_to_one")
    print(f"Merged df shape: {df.shape}")
    if len(df) == 0:
        raise RuntimeError("Merge resultou em 0 linhas.")

    df = stratified_sample_by_month(df, max_rows=MAX_ROWS_CLUSTERING)
    df = reduce_memory(df)

    thr = float(df["p_churn"].quantile(0.95))
    df["pred_action"] = (df["p_churn"] >= thr).astype("int8")
    df["fp"] = ((df["pred_action"] == 1) & (df[TARGET] == 0)).astype("int8")
    df["fn"] = ((df["pred_action"] == 0) & (df[TARGET] == 1)).astype("int8")

    feature_cols = pick_numeric_features(df)
    print(f"n_features_for_clustering: {len(feature_cols)}")
    print("features:", feature_cols)

    X = df[feature_cols].copy()

    print("\n[1/5] Pré-processamento...")
    pre_pipe = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler()),
    ])
    X_scaled = pre_pipe.fit_transform(X).astype(np.float32, copy=False)
    print(f"X_scaled shape: {X_scaled.shape} | dtype: {X_scaled.dtype}")

    print("\n[2/5] Escolhendo k via Elbow + Silhouette...")
    best_k = choose_k(X_scaled, k_range=range(2, 11))

    print(f"\n[3/5] Clustering final com k={best_k}...")
    km_final = MiniBatchKMeans(n_clusters=best_k, random_state=42, batch_size=4096, n_init=5)
    df["cluster"] = km_final.fit_predict(X_scaled)

    np.random.seed(42)
    sil_idx   = np.random.choice(len(X_scaled), min(SILHOUETTE_SAMPLE, len(X_scaled)), replace=False)
    sil_final = silhouette_score(X_scaled[sil_idx], df["cluster"].values[sil_idx], random_state=42)
    print(f"  Silhouette final (k={best_k}): {sil_final:.4f}")
    print("  Interpretação: >0.20 = estrutura razoável | >0.50 = estrutura forte")

    print("\n[4/5] Perfis dos clusters...")
    prof  = cluster_profile(df)
    drv   = simple_drivers(df, feature_cols)
    drift = monthly_drift(df)

    prof.to_csv(OUT_DIR / "unsup_cluster_profile.csv", index=False)
    drv.to_csv(OUT_DIR / "unsup_cluster_drivers.csv", index=False)
    drift.to_csv(OUT_DIR / "unsup_monthly_drift.csv", index=False)
    pd.DataFrame([{
        "k": best_k,
        "silhouette": sil_final,
        "rows_used_for_clustering": len(df),
        "max_rows_clustering": MAX_ROWS_CLUSTERING,
    }]).to_csv(OUT_DIR / "unsup_silhouette.csv", index=False)

    print("\nTop clusters por p_churn_mean:")
    print(prof.head(10).to_string(index=False))

    print("\n[5/5] Gráficos...")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = range(len(prof))
    ax.bar([i - 0.18 for i in x], prof["p_churn"] * 100, width=0.36, label="Risco previsto")
    ax.bar([i + 0.18 for i in x], prof[TARGET] * 100, width=0.36, label="Churn real")
    ax.set_xticks(list(x))
    ax.set_xticklabels(prof["cluster"])
    ax.set_xlabel("Cluster")
    ax.set_ylabel("%")
    ax.set_title(f"Clusters (k={best_k}): risco previsto × churn real\nSilhouette={sil_final:.3f}", fontsize=12)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "01_cluster_risk_vs_churn.png")

    top_drv = drv.head(10).copy().sort_values("separation_score")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(top_drv["feature"], top_drv["separation_score"])
    ax.set_xlabel("Separation score")
    ax.set_title("Top drivers dos clusters", fontsize=12)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save_fig(fig, "02_cluster_drivers.png")

    focus = drift[drift["cluster"].isin(sorted(df["cluster"].unique())[:3])].copy()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for c in sorted(focus["cluster"].unique()):
        d = focus[focus["cluster"] == c]
        ax.plot(d["safra"].astype(str), d["share"] * 100, marker="o", label=f"Cluster {c}")
    ax.set_xlabel("Safra")
    ax.set_ylabel("% da base")
    ax.set_title("Drift temporal: share dos clusters ao longo do tempo", fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "03_cluster_drift.png")

    print(f"\nOK → {OUT_DIR.resolve()}")
    print("Observação: clustering executado sobre amostra estratificada por safra "
          f"de até {MAX_ROWS_CLUSTERING:,} linhas para evitar estouro de memória.")
    print("Observação: clusters devem ser interpretados como segmentos exploratórios,")
    print("não como perfis definitivos nem evidência causal.")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        main()
