from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(".").resolve()

OUT = ROOT / "audit_pack"
OUT.mkdir(exist_ok=True)

def save_csv_if_exists(src, dst_name, n=5000):
    p = ROOT / src
    if not p.exists():
        return False
    if p.is_file() and p.suffix == ".parquet":
        df = pd.read_parquet(p)
        df.head(n).to_csv(OUT / dst_name, index=False)
        meta = {
            "source": str(p),
            "rows": int(len(df)),
            "cols": int(df.shape[1]),
            "columns": list(df.columns),
        }
        (OUT / f"{dst_name}.meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        return True
    return False

summary = {}

# 1) label trust
p = ROOT / "reports" / "label_trust_by_month.csv"
if p.exists():
    d = pd.read_csv(p)
    d.to_csv(OUT / "label_trust_by_month.csv", index=False)
    summary["label_trust"] = {
        "rows": int(len(d)),
        "months": [int(d["safra"].min()), int(d["safra"].max())] if "safra" in d.columns and len(d) else None,
        "suspect_months": d.loc[d.get("label_trust", 1) == 0, "safra"].tolist() if "label_trust" in d.columns else []
    }

# 2) scores_test completo se for pequeno; se não, amostra + métricas
scores_path = ROOT / "data" / "processed" / "scores_test.parquet"
if scores_path.exists():
    s = pd.read_parquet(scores_path)
    s.head(20000).to_csv(OUT / "scores_test_sample.csv", index=False)

    if {"churn_3m", "p_churn"}.issubset(s.columns):
        y = s["churn_3m"].astype(int)
        p = s["p_churn"].to_numpy()

        def recall_at_k(frac):
            k = max(1, int(len(s) * frac))
            idx = np.argsort(-p)[:k]
            total_pos = int(y.sum())
            captured = int(y.iloc[idx].sum())
            churn_rate = float(y.iloc[idx].mean())
            return {
                "k": k,
                "recall": 0.0 if total_pos == 0 else captured / total_pos,
                "churn_rate": churn_rate
            }

        summary["scores_test"] = {
            "rows": int(len(s)),
            "cols": int(s.shape[1]),
            "base_churn": float(y.mean()),
            "top_1": recall_at_k(0.01),
            "top_3": recall_at_k(0.03),
            "top_5": recall_at_k(0.05),
            "top_10": recall_at_k(0.10),
            "top_20": recall_at_k(0.20),
            "columns": list(s.columns),
        }

# 3) model_table: só metadados + amostra
model_path = ROOT / "data" / "processed" / "model_table"
if model_path.exists():
    m = pd.read_parquet(model_path)
    m.head(10000).to_csv(OUT / "model_table_sample.csv", index=False)
    summary["model_table"] = {
        "rows": int(len(m)),
        "cols": int(m.shape[1]),
        "columns": list(m.columns),
        "safra_min": int(pd.to_numeric(m["safra"], errors="coerce").min()) if "safra" in m.columns else None,
        "safra_max": int(pd.to_numeric(m["safra"], errors="coerce").max()) if "safra" in m.columns else None,
    }

# 4) salvar resumo
(OUT / "audit_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("Pacote gerado em:", OUT)
for f in sorted(OUT.iterdir()):
    print("-", f.name, f.stat().st_size, "bytes")