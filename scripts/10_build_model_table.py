from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEAT_LAG_DIR   = PROJECT_ROOT / "data" / "processed" / "features_lag"      # pasta particionada por safra
FEAT_BASE_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"  # status mensal
OUT_DIR        = PROJECT_ROOT / "data" / "processed" / "model_table"       # saída (dataset)

OUT_DIR.mkdir(parents=True, exist_ok=True)
(REPORTS_DIR := (PROJECT_ROOT / "reports")).mkdir(parents=True, exist_ok=True)

def add_months(yyyymm: int, n: int) -> int:
    y = yyyymm // 100
    m = yyyymm % 100
    m += n
    y += (m - 1) // 12
    m = ((m - 1) % 12) + 1
    return y * 100 + m

def read_status_month(dataset: ds.Dataset, month: int) -> pd.DataFrame:
    tbl = dataset.to_table(
        columns=["msno", "safra", "is_ativo"],
        filter=(ds.field("safra") == month)
    )
    d = tbl.to_pandas()
    d["msno"] = d["msno"].astype("string")
    d["safra"] = pd.to_numeric(d["safra"], errors="coerce").astype("int32")
    d["is_ativo"] = pd.to_numeric(d["is_ativo"], errors="coerce").fillna(0).astype("int8")
    return d.drop_duplicates(["msno", "safra"])

def main():
    base_ds = ds.dataset(str(FEAT_BASE_PATH), format="parquet")

    months = sorted([
        int(p.name.split("=")[1])
        for p in FEAT_LAG_DIR.iterdir()
        if p.is_dir() and p.name.startswith("safra=")
    ])
    if not months:
        raise FileNotFoundError(f"Não encontrei partições safra=... em {FEAT_LAG_DIR}")

    min_m, max_m = months[0], months[-1]
    print(f"Meses features_lag: {min_m} -> {max_m} | n_months={len(months)}")

    # -------------------------
    # 1) Diagnóstico por mês (label_trust)
    # -------------------------
    diag_rows = []
    for m in tqdm(months, desc="diag_by_month"):
        fut_m = add_months(m, 3)
        if fut_m > max_m:
            continue

        st_now = read_status_month(base_ds, m)
        st_fut = read_status_month(base_ds, fut_m)[["msno", "is_ativo"]].rename(columns={"is_ativo": "is_ativo_fut"})

        join = st_now.merge(st_fut, on="msno", how="left")
        obs = join[join["is_ativo_fut"].notna()].copy()

        if len(obs) == 0:
            diag_rows.append({"safra": m, "rows": 0, "fut_active_rate": 0.0, "churn_rate": 0.0, "label_trust": 0})
            continue

        elig = obs[obs["is_ativo"] == 1].copy()
        if len(elig) == 0:
            diag_rows.append({
                "safra": m,
                "rows": int(len(obs)),
                "fut_active_rate": float(obs["is_ativo_fut"].mean()),
                "churn_rate": 0.0,
                "label_trust": 0
            })
            continue

        churn = ((elig["is_ativo"] == 1) & (elig["is_ativo_fut"] == 0)).astype("int8")
        fut_active_rate = float(obs["is_ativo_fut"].mean())
        churn_rate = float(churn.mean())

        # regra B (censoring)
        label_trust = 0 if (fut_active_rate < 0.01 or churn_rate > 0.95) else 1
        diag_rows.append({
            "safra": m,
            "rows": int(len(elig)),
            "fut_active_rate": fut_active_rate,
            "churn_rate": churn_rate,
            "label_trust": int(label_trust),
        })

    diag = pd.DataFrame(diag_rows)
    diag_path = REPORTS_DIR / "label_trust_by_month.csv"
    diag.to_csv(diag_path, index=False)

    print("OK ->", diag_path)
    print("Suspicious months (label_trust=0):")
    print(diag[diag["label_trust"] == 0][["safra","rows","fut_active_rate","churn_rate","label_trust"]].sort_values("safra").to_string(index=False))

    # -------------------------
    # 2) Escrever model_table mês a mês (dataset)
    # -------------------------
    for m in tqdm(months, desc="write_model_table"):
        fut_m = add_months(m, 3)
        if fut_m > max_m:
            continue

        lag_part = FEAT_LAG_DIR / f"safra={m}"
        if not lag_part.exists():
            continue

        feat = pd.read_parquet(lag_part)
        feat["msno"] = feat["msno"].astype("string")

        # IMPORTANT: ao ler partição, 'safra' pode não vir como coluna -> injeta
        feat["safra"] = np.int32(m)

        st_now = read_status_month(base_ds, m)[["msno","is_ativo"]]
        st_fut = read_status_month(base_ds, fut_m)[["msno","is_ativo"]].rename(columns={"is_ativo":"is_ativo_fut"})

        df = feat.merge(st_now, on="msno", how="left").merge(st_fut, on="msno", how="left")
        df["is_ativo"] = pd.to_numeric(df["is_ativo"], errors="coerce").fillna(0).astype("int8")

        # rotula só onde futuro observado
        df = df[df["is_ativo_fut"].notna()].copy()
        df["is_ativo_fut"] = pd.to_numeric(df["is_ativo_fut"], errors="coerce").fillna(0).astype("int8")

        df["churn_3m"] = ((df["is_ativo"] == 1) & (df["is_ativo_fut"] == 0)).astype("int8")

        row = diag[diag["safra"] == m]
        if len(row) == 0:
            df["label_trust"] = np.int8(0)
            df["fut_active_rate"] = np.nan
            df["churn_rate_month"] = np.nan
        else:
            df["label_trust"] = np.int8(int(row["label_trust"].iloc[0]))
            df["fut_active_rate"] = float(row["fut_active_rate"].iloc[0])
            df["churn_rate_month"] = float(row["churn_rate"].iloc[0])

        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_to_dataset(table, root_path=str(OUT_DIR), partition_cols=["safra"])

    print("OK ->", OUT_DIR)
    print("Obs: model_table é um dataset parquet (pasta), não um arquivo único.")

if __name__ == "__main__":
    main()