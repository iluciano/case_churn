from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_TABLE_DIR = PROJECT_ROOT / "data" / "processed" / "model_table"
SCORES_TEST = PROJECT_ROOT / "data" / "processed" / "scores_test.parquet"
OUT_DIR = PROJECT_ROOT / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "churn_3m"

def pick_numeric_features(df: pd.DataFrame) -> list[str]:
    drop = {
        "safra", "safra_fut", "msno", TARGET,
        "p_churn", "is_ativo", "is_ativo_fut",
        "pred_action", "fp", "fn",
        "label_trust", "fut_active_rate", "churn_rate_month",
    }
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in drop]

    preferred = [
        "num_unq", "total_secs", "plays_total",
        "paid_sum", "auto_renew_rate", "cancel_txn_count",
        "has_usage", "has_payment",
        "total_secs_lag1", "total_secs_diff1",
        "num_unq_lag1", "num_unq_diff1",
        "paid_sum_lag1", "paid_sum_diff1",
        "plays_total_lag1", "plays_total_diff1",
        "has_usage_lag1", "has_usage_diff1",
        "has_payment_lag1", "has_payment_diff1",
    ]
    cols = [c for c in preferred if c in num_cols]
    if not cols:
        cols = num_cols
    if len(cols) > 40:
        cols = cols[:40]
    return cols

def cluster_profile(df: pd.DataFrame, label_col: str = "cluster") -> pd.DataFrame:
    agg = {"msno": "count", "p_churn": "mean", TARGET: "mean", "fp": "mean", "fn": "mean"}
    for c in ["paid_sum", "total_secs", "num_unq", "auto_renew_rate", "cancel_txn_count"]:
        if c in df.columns:
            agg[c] = "mean"
    out = df.groupby(label_col).agg(agg).rename(columns={"msno": "rows"}).reset_index()
    out["share"] = out["rows"] / out["rows"].sum()
    out = out.sort_values("p_churn", ascending=False)
    return out

def simple_drivers(df: pd.DataFrame, feature_cols: list[str], label_col: str = "cluster", top_n: int = 15) -> pd.DataFrame:
    eps = 1e-9
    overall_var = df[feature_cols].var(numeric_only=True)
    rows = []
    for c in feature_cols:
        means = df.groupby(label_col)[c].mean()
        score = float(means.var() / (overall_var.get(c, eps) + eps))
        rows.append({"feature": c, "separation_score": score})
    return pd.DataFrame(rows).sort_values("separation_score", ascending=False).head(top_n)

def monthly_drift(df: pd.DataFrame, label_col: str = "cluster") -> pd.DataFrame:
    g = df.groupby(["safra", label_col]).agg(
        rows=("msno", "count"),
        p_churn_mean=("p_churn", "mean"),
        churn_rate=(TARGET, "mean"),
    ).reset_index()
    total = g.groupby("safra")["rows"].transform("sum")
    g["share"] = g["rows"] / total
    return g.sort_values(["safra", label_col])

def main():
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("OUT_DIR:", OUT_DIR.resolve())

    if not MODEL_TABLE_DIR.exists():
        raise FileNotFoundError(f"Não encontrei {MODEL_TABLE_DIR}.")
    if not SCORES_TEST.exists():
        raise FileNotFoundError(f"Não encontrei {SCORES_TEST}.")

    model = pd.read_parquet(MODEL_TABLE_DIR)
    scores = pd.read_parquet(SCORES_TEST)

    # evita colisão: removemos churn_3m do model (scores é a fonte de truth do TEST)
    model = model.drop(columns=[TARGET], errors="ignore")

    df = model.merge(scores[["msno", "safra", "p_churn", TARGET]], on=["msno", "safra"], how="inner")
    print("merged df shape:", df.shape)

    if len(df) == 0:
        raise RuntimeError("Merge resultou em 0 linhas.")

    # Política de campanha: top 5%
    thr = df["p_churn"].quantile(0.95)
    df["pred_action"] = (df["p_churn"] >= thr).astype(int)
    df["fp"] = ((df["pred_action"] == 1) & (df[TARGET] == 0)).astype(int)
    df["fn"] = ((df["pred_action"] == 0) & (df[TARGET] == 1)).astype(int)

    feature_cols = pick_numeric_features(df)
    if not feature_cols:
        raise RuntimeError("Não encontrei features numéricas para clustering.")

    print("n_features_for_clustering:", len(feature_cols))

    X = df[feature_cols].copy()
    pipe = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("km", MiniBatchKMeans(n_clusters=6, random_state=42, batch_size=8192)),
    ])
    df["cluster"] = pipe.fit_predict(X)

    prof = cluster_profile(df, label_col="cluster")
    drv = simple_drivers(df, feature_cols, label_col="cluster", top_n=15)
    drift = monthly_drift(df, label_col="cluster")

    prof_path = OUT_DIR / "unsup_cluster_profile.csv"
    drv_path = OUT_DIR / "unsup_cluster_drivers.csv"
    drift_path = OUT_DIR / "unsup_monthly_drift.csv"

    prof.to_csv(prof_path, index=False)
    drv.to_csv(drv_path, index=False)
    drift.to_csv(drift_path, index=False)

    print("WROTE:", prof_path.resolve())
    print("WROTE:", drv_path.resolve())
    print("WROTE:", drift_path.resolve())
    print("\nTOP clusters by p_churn_mean:")
    print(prof.head(10).to_string(index=False))

if __name__ == "__main__":
    main()