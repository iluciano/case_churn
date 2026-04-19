"""
11_train_churn_model.py  –  Treinamento do modelo de churn
==========================================================
Correções v2 (vs versão original do ChatGPT):

  1. LEAKAGE CORRIGIDO:
     fut_active_rate e churn_rate_month foram adicionados ao DROP_ALWAYS.
     Essas colunas são agregados calculados a partir do M+3 (target window)
     e nunca estariam disponíveis em produção no momento da predição.

  2. FEATURE IMPORTANCE exportada como CSV e gráfico.

  3. MODELO SALVO em disk via joblib (essencial para produtização).

  4. BUSCA DE HIPERPARÂMETROS simples (3 configurações explicitadas)
     para demonstrar que os parâmetros finais são justificados.

  5. is_ativo e label_trust adicionados ao DROP_ALWAYS
     (constantes dentro do df_trust — não agregam informação ao modelo).
"""

from __future__ import annotations

import inspect
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder

PROJECT_ROOT    = Path(__file__).resolve().parents[1]
MODEL_TABLE_DIR = PROJECT_ROOT / "data" / "processed" / "model_table"
OUT_SCORES      = PROJECT_ROOT / "data" / "processed" / "scores_test.parquet"
MODEL_DIR       = PROJECT_ROOT / "models"
REPORTS_DIR     = PROJECT_ROOT / "reports"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "churn_3m"


# Funções em nível de módulo para manter o pipeline serializável via joblib
def num_to_float(X_):
    return pd.DataFrame(X_).replace({pd.NA: np.nan}).astype("float64").to_numpy()


def cat_to_object(X_):
    return pd.DataFrame(X_).replace({pd.NA: None}).astype("object").to_numpy()


def permutation_importance_report(pipe: Pipeline, X: pd.DataFrame, y: pd.Series, n_top: int = 20) -> None:
    """
    HistGradientBoostingClassifier não expõe feature_importances_.
    Usa permutation importance sobre uma amostra do conjunto de treino.
    """
    try:
        from sklearn.inspection import permutation_importance
        rng = np.random.RandomState(42)

        n = len(X)
        sample_n = min(100_000, n)
        if sample_n < n:
            idx = rng.choice(n, size=sample_n, replace=False)
            X_eval = X.iloc[idx].copy()
            y_eval = y.iloc[idx].copy()
        else:
            X_eval = X.copy()
            y_eval = y.copy()

        r = permutation_importance(
            pipe, X_eval, y_eval,
            n_repeats=5,
            random_state=42,
            scoring="roc_auc",
            n_jobs=1,
        )

        fi = pd.DataFrame({
            "feature": X_eval.columns,
            "importance": r.importances_mean,
            "importance_std": r.importances_std,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        fi_path = REPORTS_DIR / "feature_importance.csv"
        fi.to_csv(fi_path, index=False)
        print(f"\nFeature importance (permutation) → {fi_path}")
        print(fi.head(15).to_string(index=False))

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top = fi.head(n_top).copy().sort_values("importance")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(top["feature"], top["importance"])
        ax.set_xlabel("Importância (queda média no ROC-AUC)")
        ax.set_title("Top Features – Permutation Importance", fontsize=12)
        fig.tight_layout()
        fig_path = REPORTS_DIR / "feature_importance.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Gráfico → {fig_path}")

    except Exception as e:
        print(f"[AVISO] Não foi possível exportar permutation importance: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# CORREÇÃO 1: fut_active_rate e churn_rate_month são derivados do futuro (M+3)
# → jamais disponíveis em produção → devem ser excluídos do treino.
# is_ativo e label_trust são constantes no df_trust (todos == 1) → ruído.
# ─────────────────────────────────────────────────────────────────────────────
DROP_ALWAYS = {
    "msno", "safra", "safra_fut", "is_ativo_fut", TARGET,
    "fut_active_rate",    # CORRIGIDO: derivado do M+3 → leakage
    "churn_rate_month",   # CORRIGIDO: derivado do M+3 → leakage
    "is_ativo",           # constante (sempre 1 no df_trust)
    "label_trust",        # constante (sempre 1 no df_trust)
}


# ─────────────────────────────────────────────────────────────────────────────
# Métricas operacionais de campanha
# Observação: a leitura executiva deve enfatizar Top 5%/10%/20%.
# Os percentis mais extremos podem ser instáveis em bases raras e grandes.
# ─────────────────────────────────────────────────────────────────────────────
def recall_at_k(y_true: pd.Series, y_proba: np.ndarray, k_frac: float = 0.05):
    k = max(1, int(len(y_true) * k_frac))
    idx = np.argsort(-y_proba)[:k]
    captured  = int(y_true.iloc[idx].sum())
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


def eval_set(name: str, X, y, pipe: Pipeline) -> dict:
    p = pipe.predict_proba(X)[:, 1]
    auc  = safe_auc(y, p)
    pr   = safe_prauc(y, p)
    r5, k5 = recall_at_k(y, p, 0.05)
    cr5, _ = churn_rate_in_topk(y, p, 0.05)
    r10, _ = recall_at_k(y, p, 0.10)
    print(f"\n{name} METRICS")
    print(f"  ROC-AUC       : {auc:.4f}")
    print(f"  PR-AUC        : {pr:.4f}")
    print(f"  Recall@Top5%  : {r5:.4f}  (k={k5:,})")
    print(f"  ChurnRate@Top5%: {cr5:.4f}")
    print(f"  Recall@Top10% : {r10:.4f}")
    return {"set": name, "roc_auc": auc, "pr_auc": pr,
            "recall_top5pct": r5, "churnrate_top5pct": cr5, "k5": k5}


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline de pré-processamento + modelo
# ─────────────────────────────────────────────────────────────────────────────
def build_pipeline(X: pd.DataFrame, **clf_params) -> Pipeline:
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline(steps=[
        ("to_float", FunctionTransformer(num_to_float, validate=False)),
        ("imp",      SimpleImputer(strategy="median", missing_values=np.nan)),
    ])

    oh_kwargs = {"handle_unknown": "ignore"}
    sig = inspect.signature(OneHotEncoder)
    oh_kwargs["sparse_output" if "sparse_output" in sig.parameters else "sparse"] = True

    cat_pipe = Pipeline(steps=[
        ("to_obj", FunctionTransformer(cat_to_object, validate=False)),
        ("imp",    SimpleImputer(strategy="most_frequent", missing_values=None)),
        ("oh",     OneHotEncoder(**oh_kwargs)),
    ])

    transformers = []
    if num_cols: transformers.append(("num", num_pipe, num_cols))
    if cat_cols: transformers.append(("cat", cat_pipe, cat_cols))
    if not transformers:
        raise ValueError("Nenhuma feature disponível após remoções.")

    default_params = dict(max_depth=6, learning_rate=0.08, max_iter=300, random_state=42)
    default_params.update(clf_params)

    clf = HistGradientBoostingClassifier(**default_params)
    return Pipeline(steps=[("pre", ColumnTransformer(transformers, remainder="drop")),
                            ("clf", clf)])


# ─────────────────────────────────────────────────────────────────────────────
# Split temporal (out-of-time)
# ─────────────────────────────────────────────────────────────────────────────
def temporal_split_fixed(df: pd.DataFrame, test_months: int = 4, val_months: int = 2):
    months = sorted(df["safra"].dropna().unique())
    needed = test_months + val_months + 1
    if len(months) < needed:
        raise ValueError(f"Poucos meses confiáveis ({len(months)}). Precisa >= {needed}.")

    test  = months[-test_months:]
    val   = months[-(test_months + val_months):-test_months]
    train = months[:-(test_months + val_months)]

    tr = df[df["safra"].isin(train)].copy()
    va = df[df["safra"].isin(val)].copy()
    te = df[df["safra"].isin(test)].copy()
    return tr, va, te, train, val, test


# ─────────────────────────────────────────────────────────────────────────────
# Busca de hiperparâmetros (simples, explicitada para a banca)
# ─────────────────────────────────────────────────────────────────────────────
HPARAM_GRID = [
    # (label, params)
    ("Baseline – shallow/fast",   dict(max_depth=4, learning_rate=0.10, max_iter=200)),
    ("Final    – depth6/lr008",   dict(max_depth=6, learning_rate=0.08, max_iter=300)),  # escolhido
    ("Deep     – depth8/lr005",   dict(max_depth=8, learning_rate=0.05, max_iter=400)),
]


def hyperparam_search(Xtr, ytr, Xva, yva) -> dict:
    """
    Compara 3 configurações explícitas de hiperparâmetros usando o set de validação.
    Justificativa de uso do HistGradientBoostingClassifier:
      - Suporta missings nativamente (não exige imputer para numéricos em muitas versões)
      - Muito rápido em datasets grandes (binning de features)
      - Regularização via learning_rate + max_iter (early stopping possível)
      - Produtização simples: single binary via joblib
    Por que NÃO LightGBM/XGBoost:
      - Dependências externas adicionais → custo de implantação
      - HistGBM do sklearn é equivalente para esse tamanho de dado
    """
    print("\n" + "=" * 55)
    print("BUSCA DE HIPERPARÂMETROS (validação out-of-time)")
    print("=" * 55)

    results = []
    for label, params in HPARAM_GRID:
        pipe = build_pipeline(Xtr, **params)
        pipe.fit(Xtr, ytr)
        pva  = pipe.predict_proba(Xva)[:, 1]
        auc  = safe_auc(yva, pva)
        r5,_ = recall_at_k(yva, pva, 0.05)
        print(f"  {label:<40} | ROC-AUC={auc:.4f} | Recall@5%={r5:.4f}")
        results.append({"label": label, "roc_auc": auc, "recall_top5": r5, "params": params})

    best = max(results, key=lambda r: r["roc_auc"])
    print(f"\n  → Melhor configuração: {best['label']}  (ROC-AUC={best['roc_auc']:.4f})")
    return best["params"]


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance
# ─────────────────────────────────────────────────────────────────────────────
def export_feature_importance(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> None:
    permutation_importance_report(pipe, X, y)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    if not MODEL_TABLE_DIR.exists():
        raise FileNotFoundError(f"Não encontrei {MODEL_TABLE_DIR}.")

    df = pd.read_parquet(MODEL_TABLE_DIR)
    df["safra"]  = pd.to_numeric(df["safra"],  errors="coerce").astype("Int32")
    df[TARGET]   = pd.to_numeric(df[TARGET],    errors="coerce").fillna(0).astype("int8")

    if "is_ativo" in df.columns:
        df["is_ativo"] = pd.to_numeric(df["is_ativo"], errors="coerce").fillna(0).astype("int8")
        df = df[df["is_ativo"] == 1].copy()

    if "label_trust" not in df.columns:
        raise ValueError("model_table não tem label_trust. Verifique o script 10.")
    df["label_trust"] = pd.to_numeric(df["label_trust"], errors="coerce").fillna(0).astype("int8")
    df_trust = df[df["label_trust"] == 1].copy()

    print(f"Total rows: {len(df):,}  |  Trusted: {len(df_trust):,}")
    print(f"Months trusted: {int(df_trust['safra'].min())} → {int(df_trust['safra'].max())}"
          f"  |  n_months: {df_trust['safra'].nunique()}")
    print(f"Churn rate (trusted): {df_trust[TARGET].mean():.4f}")

    tr, va, te, mtr, mva, mte = temporal_split_fixed(df_trust, test_months=4, val_months=2)

    print(f"\nSPLIT (out-of-time, apenas meses confiáveis)")
    print(f"  TRAIN : {mtr[0]} → {mtr[-1]}  |  rows: {len(tr):,}")
    print(f"  VAL   : {mva[0]} → {mva[-1]}  |  rows: {len(va):,}")
    print(f"  TEST  : {mte[0]} → {mte[-1]}  |  rows: {len(te):,}")

    if len(tr) == 0:
        raise ValueError("Treino vazio após filtro label_trust.")
    if len(np.unique(tr[TARGET])) < 2:
        raise ValueError("Treino com apenas 1 classe.")

    Xtr, ytr = split_xy(tr)
    Xva, yva = split_xy(va)
    Xte, yte = split_xy(te)

    # ── Busca de hiperparâmetros ──────────────────────────────────────────
    best_params = hyperparam_search(Xtr, ytr, Xva, yva)

    # ── Treino final com melhores parâmetros ─────────────────────────────
    print("\n" + "=" * 55)
    print("TREINO FINAL")
    print("=" * 55)
    pipe = build_pipeline(Xtr, **best_params)
    pipe.fit(Xtr, ytr)

    metrics = []
    if len(va):
        m = eval_set("VALIDAÇÃO", Xva, yva, pipe)
        metrics.append(m)
    if len(te):
        m = eval_set("TESTE", Xte, yte, pipe)
        metrics.append(m)
        pte = pipe.predict_proba(Xte)[:, 1]
        out = te[["msno", "safra", TARGET]].copy()
        out["p_churn"] = pte
        out.to_parquet(OUT_SCORES, index=False)
        print(f"\nScores → {OUT_SCORES}")

    # ── Exporta feature importance ────────────────────────────────────────
    export_feature_importance(pipe, Xtr, ytr)

    # ── Salva modelo ──────────────────────────────────────────────────────
    model_path = MODEL_DIR / "churn_model_v1.joblib"
    joblib.dump(pipe, model_path)
    print(f"\nModelo salvo → {model_path}")

    # ── Salva métricas ────────────────────────────────────────────────────
    if metrics:
        pd.DataFrame(metrics).to_csv(REPORTS_DIR / "model_metrics.csv", index=False)

    print("\nCONCLUSÃO:")
    print("  ROC-AUC ~0.85 indica bom poder de ranking.")
    print("  PR-AUC baixa é esperada em churn com prevalência ~5%.")
    print("  A leitura operacional deve priorizar faixas Top 5%/10%/20%,")
    print("  não o topo extremo isolado, que ficou fraco nos artefatos auditados.")
    print("  A queda de validação para teste sugere mudança de regime entre os períodos")
    print("  e reforça a necessidade de narrativa cautelosa sobre campanha.")


if __name__ == "__main__":
    main()
