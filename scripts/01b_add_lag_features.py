from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INP = PROJECT_ROOT / "data" / "processed" / "features.parquet"

# Vamos gravar como dataset (pasta) para suportar escrita incremental
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "features_lag"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Colunas que vamos criar lag1/diff1 (mantém enxuto)
BASE_COLS = [
    "total_secs", "num_unq", "paid_sum", "plays_total",
    "has_usage", "has_payment", "auto_renew_rate", "cancel_txn_count"
]

def add_1_month(yyyymm: int) -> int:
    y = yyyymm // 100
    m = yyyymm % 100
    m += 1
    if m == 13:
        y += 1
        m = 1
    return y * 100 + m

def main():
    if not INP.exists():
        raise FileNotFoundError(f"Não encontrei {INP}. Rode antes o scripts/01_build_features.py")

    data = ds.dataset(str(INP), format="parquet")
    # Descobre meses disponíveis sem carregar tudo
    safra_vals = data.to_table(columns=["safra"]).column("safra").to_pylist()
    # safra_vals pode ser grande; então vamos usar pyarrow compute unique
    safra_unique = ds.dataset(str(INP), format="parquet").to_table(columns=["safra"]).column("safra").unique().to_pylist()
    months = sorted([int(x) for x in safra_unique if x is not None])

    # mantemos apenas colunas necessárias + chaves
    wanted_cols = ["msno", "safra"] + [c for c in BASE_COLS if c in data.schema.names]
    # garante existência
    wanted_cols = [c for c in wanted_cols if c in data.schema.names]

    print("Meses:", months[0], "->", months[-1], "| n_months:", len(months))
    print("Cols:", wanted_cols)

    # cache do mês anterior: msno -> valores base
    prev = None
    prev_month = None

    for month in tqdm(months, desc="lag_by_month"):
        # lê só o mês corrente
        cur_tbl = data.to_table(columns=wanted_cols, filter=(ds.field("safra") == month))
        cur = cur_tbl.to_pandas()
        cur["msno"] = cur["msno"].astype("string")
        cur["safra"] = pd.to_numeric(cur["safra"], errors="coerce").astype("int32")

        # prepara colunas base como numéricas
        for c in BASE_COLS:
            if c in cur.columns:
                cur[c] = pd.to_numeric(cur[c], errors="coerce")

        if prev is None:
            # sem lag no primeiro mês
            for c in BASE_COLS:
                if c in cur.columns:
                    cur[f"{c}_lag1"] = np.nan
                    cur[f"{c}_diff1"] = np.nan
            out_tbl = pa.Table.from_pandas(cur, preserve_index=False)
            pq.write_to_dataset(out_tbl, root_path=str(OUT_DIR), partition_cols=["safra"])
            prev = cur[["msno"] + [c for c in BASE_COLS if c in cur.columns]].copy()
            prev_month = month
            continue

        # Só cria lag se o mês anterior for exatamente month-1
        # (se tiver buracos, lag fica NaN)
        expected_prev = month - 1
        # mas como YYYYMM não é linear, usamos função
        # aqui vamos checar se prev_month é o mês imediatamente anterior
        # gerando month anterior a partir do prev_month e comparando:
        is_consecutive = (add_1_month(prev_month) == month)

        if is_consecutive:
            # join msno com prev
            cur = cur.merge(prev, on="msno", how="left", suffixes=("", "_prev"))
            for c in BASE_COLS:
                if c in cur.columns and f"{c}_prev" in cur.columns:
                    cur[f"{c}_lag1"] = cur[f"{c}_prev"]
                    cur[f"{c}_diff1"] = cur[c] - cur[f"{c}_prev"]
                    cur.drop(columns=[f"{c}_prev"], inplace=True)
        else:
            for c in BASE_COLS:
                if c in cur.columns:
                    cur[f"{c}_lag1"] = np.nan
                    cur[f"{c}_diff1"] = np.nan

        out_tbl = pa.Table.from_pandas(cur, preserve_index=False)
        pq.write_to_dataset(out_tbl, root_path=str(OUT_DIR), partition_cols=["safra"])

        # atualiza cache
        prev = cur[["msno"] + [c for c in BASE_COLS if c in cur.columns]].copy()
        prev_month = month

    print("OK ->", OUT_DIR)
    print("Obs: saída é um dataset parquet particionado por safra (pasta).")

if __name__ == "__main__":
    main()