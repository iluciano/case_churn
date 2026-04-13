from __future__ import annotations

from pathlib import Path
import inspect
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import HistGradientBoostingClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_TABLE_DIR = PROJECT_ROOT / "data" / "processed" / "model_table"   # PASTA (dataset parquet)
OUT_SCORES = PROJECT_ROOT / "data" / "processed" / "scores_test.parquet"

TARGET = "churn_3m"
DROP_ALWAYS = {"msno", "safra", "safra_fut", "is_ativo_fut", TARGET}


# -----------------------------
# Métricas para campanha
# -----------------------------
def recall_at_k(y_true: pd.Series, y_proba: np.ndarray, k_frac: float = 0.05):
    k = max(1, int(len(y_true) * k_frac))
    idx = np.argsort(-y_proba)[:k]
    captured = int(y_true.iloc[idx].sum())
    total_pos = int(y_true.sum())
    return (captured / total_pos) if total_pos else 0.0, k


def churn_rate_in_topk(y_true: pd.Series, y_proba: np.ndarray, k_frac: float = 0.05):
    k = max(1, int(len(y_true) * k_frac))
    idx = np.argsort(-y_proba)[:k]
    return float(y_true.iloc[idx].mean()), k


def safe_auc(y_true, y_proba):
    return np.nan if len(np.unique(y_true)) < 2 else roc_auc_score(y_true, y_proba)


def safe_prauc(y_true, y_proba):
    return np.nan if len(np.unique(y_true)) < 2 else average_precision_score(y_true, y_proba)


def split_xy(d: pd.DataFrame):
    y = d[TARGET].astype(int)
    X = d.drop(columns=[c for c in DROP_ALWAYS if c in d.columns], errors="ignore")
    return X, y


# -----------------------------
# Pipeline robusto:
# - numéricas = somente numéricas reais
# - categóricas incluem object/string/category
# - trata pd.NA
# - onehot compatível com versões sklearn
# -----------------------------
def build_pipeline(X: pd.DataFrame) -> Pipeline:
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    def num_to_float(X_):
        df_ = pd.DataFrame(X_).replace({pd.NA: np.nan})
        return df_.astype("float64").to_numpy()

    def cat_to_object(X_):
        df_ = pd.DataFrame(X_).replace({pd.NA: None})
        return df_.astype("object").to_numpy()

    num_pipe = Pipeline(steps=[
        ("to_float", FunctionTransformer(num_to_float, validate=False)),
        ("imp", SimpleImputer(strategy="median", missing_values=np.nan)),
    ])

    oh_kwargs = {"handle_unknown": "ignore"}
    sig = inspect.signature(OneHotEncoder)
    if "sparse_output" in sig.parameters:
        oh_kwargs["sparse_output"] = True
    elif "sparse" in sig.parameters:
        oh_kwargs["sparse"] = True

    cat_pipe = Pipeline(steps=[
        ("to_obj", FunctionTransformer(cat_to_object, validate=False)),
        ("imp", SimpleImputer(strategy="most_frequent", missing_values=None)),
        ("oh", OneHotEncoder(**oh_kwargs)),
    ])

    transformers = []
    if len(num_cols) > 0:
        transformers.append(("num", num_pipe, num_cols))
    if len(cat_cols) > 0:
        transformers.append(("cat", cat_pipe, cat_cols))

    if not transformers:
        raise ValueError("Não há features para treinar após remoções.")

    pre = ColumnTransformer(transformers=transformers, remainder="drop")

    clf = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.08,
        max_iter=300,
        random_state=42,
    )

    return Pipeline(steps=[("pre", pre), ("clf", clf)])


# -----------------------------
# Split temporal fixo em meses CONFIÁVEIS
# TEST: últimos 6 meses confiáveis
# VAL : 3 meses antes do TEST
# TRAIN: resto
# -----------------------------
def temporal_split_fixed(df: pd.DataFrame, test_months: int = 6, val_months: int = 3):
    df = df.sort_values("safra").reset_index(drop=True)
    months = sorted(df["safra"].dropna().unique())

    needed = test_months + val_months + 1
    if len(months) < needed:
        raise ValueError(f"Poucos meses confiáveis ({len(months)}). Precisa >= {needed}.")

    test = months[-test_months:]
    val = months[-(test_months + val_months):-test_months]
    train = months[:-(test_months + val_months)]

    tr = df[df["safra"].isin(train)].copy()
    va = df[df["safra"].isin(val)].copy()
    te = df[df["safra"].isin(test)].copy()
    return tr, va, te, train, val, test


def month_counts(df: pd.DataFrame, months: list[int]) -> dict:
    d = df[df["safra"].isin(months)]
    vc = d[TARGET].value_counts().to_dict()
    return {0: int(vc.get(0, 0)), 1: int(vc.get(1, 0)), "rows": int(len(d))}


def main():
    if not MODEL_TABLE_DIR.exists():
        raise FileNotFoundError(f"Não encontrei {MODEL_TABLE_DIR}. Rode antes o script 10 streaming.")

    # lê dataset parquet (pasta)
    df = pd.read_parquet(MODEL_TABLE_DIR)

    # tipos
    df["safra"] = pd.to_numeric(df["safra"], errors="coerce").astype("Int32")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce").fillna(0).astype("int8")

    # elegíveis: ativo em M
    if "is_ativo" in df.columns:
        df["is_ativo"] = pd.to_numeric(df["is_ativo"], errors="coerce").fillna(0).astype("int8")
        df = df[df["is_ativo"] == 1].copy()

    # filtro label_trust
    if "label_trust" not in df.columns:
        raise ValueError("model_table não tem label_trust. Verifique o script 10.")
    df["label_trust"] = pd.to_numeric(df["label_trust"], errors="coerce").fillna(0).astype("int8")

    df_trust = df[df["label_trust"] == 1].copy()

    print("DEBUG total rows:", len(df), "| trusted rows:", len(df_trust))
    print("DEBUG months trusted:", int(df_trust["safra"].min()), "->", int(df_trust["safra"].max()),
          "| n_months:", df_trust["safra"].nunique())
    print("DEBUG churn trusted:", float(df_trust[TARGET].mean()))

    # split
    tr, va, te, mtr, mva, mte = temporal_split_fixed(df_trust, test_months=4, val_months=2)

    print("\nSPLIT (trusted only)")
    print("TRAIN:", mtr[:3], "...", mtr[-3:], month_counts(df_trust, mtr))
    print("VAL  :", mva, month_counts(df_trust, mva))
    print("TEST :", mte, month_counts(df_trust, mte))

    if len(tr) == 0:
        raise ValueError("Treino ficou vazio após filtro label_trust.")
    if len(np.unique(tr[TARGET])) < 2:
        raise ValueError("Treino tem apenas 1 classe após filtro label_trust.")

    Xtr, ytr = split_xy(tr)
    Xva, yva = split_xy(va)
    Xte, yte = split_xy(te)

    pipe = build_pipeline(Xtr)
    pipe.fit(Xtr, ytr)

    # VAL
    if len(va):
        pva = pipe.predict_proba(Xva)[:, 1]
        print("\nVAL METRICS")
        print("ROC-AUC:", safe_auc(yva, pva))
        print("PR-AUC :", safe_prauc(yva, pva))
        r5, k5 = recall_at_k(yva, pva, 0.05)
        cr5, _ = churn_rate_in_topk(yva, pva, 0.05)
        print(f"Recall@Top5% (k={k5}): {r5:.3f}")
        print(f"ChurnRate@Top5% (k={k5}): {cr5:.3f}")

    # TEST
    if len(te):
        pte = pipe.predict_proba(Xte)[:, 1]
        print("\nTEST METRICS")
        print("ROC-AUC:", safe_auc(yte, pte))
        print("PR-AUC :", safe_prauc(yte, pte))
        r5t, k5t = recall_at_k(yte, pte, 0.05)
        cr5t, _ = churn_rate_in_topk(yte, pte, 0.05)
        print(f"Recall@Top5% (k={k5t}): {r5t:.3f}")
        print(f"ChurnRate@Top5% (k={k5t}): {cr5t:.3f}")

        out = te[["msno", "safra", TARGET]].copy()
        out["p_churn"] = pte
        out.to_parquet(OUT_SCORES, index=False)
        print("\nOK ->", OUT_SCORES)


if __name__ == "__main__":
    main()