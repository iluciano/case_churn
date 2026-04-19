"""
01b_add_lag_features.py  –  Criação de features de lag e diferença temporal
============================================================================
Para cada mês M, cria:
  - {col}_lag1  : valor da coluna no mês M-1 (NaN se não consecutivo)
  - {col}_diff1 : variação M vs M-1 (NaN se não consecutivo)

Saída: data/processed/features_lag/  (dataset parquet particionado por safra)

CORREÇÃO v2:
  - Removida variável morta `safra_vals` (linha que carregava tudo na memória)
"""

from __future__ import annotations

from pathlib import Path
import shutil
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INP     = PROJECT_ROOT / "data" / "processed" / "features.parquet"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "features_lag"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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


def reset_output_dir(path: Path) -> None:
    """
    Garante idempotência do dataset particionado.
    Sem limpeza prévia, reruns acumulam arquivos parquet antigos na pasta.
    """
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main():
    if not INP.exists():
        raise FileNotFoundError(
            f"Não encontrei {INP}. Rode antes o scripts/01_build_features.py"
        )

    data = ds.dataset(str(INP), format="parquet")

    # ── Descobre meses únicos sem carregar toda a coluna na memória ──
    # CORREÇÃO: linha abaixo removida (carregava lista completa desnecessariamente):
    #   safra_vals = data.to_table(columns=["safra"]).column("safra").to_pylist()
    months = sorted([
        int(x)
        for x in data.to_table(columns=["safra"]).column("safra").unique().to_pylist()
        if x is not None
    ])

    wanted_cols = ["msno", "safra"] + [c for c in BASE_COLS if c in data.schema.names]
    wanted_cols = [c for c in wanted_cols if c in data.schema.names]

    print(f"Meses: {months[0]} → {months[-1]} | n_months: {len(months)}")
    print(f"Cols: {wanted_cols}")

    reset_output_dir(OUT_DIR)
    print(f"Diretório de saída resetado: {OUT_DIR}")

    prev       = None
    prev_month = None

    for month in tqdm(months, desc="lag_by_month"):
        cur_tbl = data.to_table(columns=wanted_cols,
                                filter=(ds.field("safra") == month))
        cur = cur_tbl.to_pandas()
        cur["msno"]  = cur["msno"].astype("string")
        cur["safra"] = pd.to_numeric(cur["safra"], errors="coerce").astype("int32")

        for c in BASE_COLS:
            if c in cur.columns:
                cur[c] = pd.to_numeric(cur[c], errors="coerce")

        if prev is None:
            for c in BASE_COLS:
                if c in cur.columns:
                    cur[f"{c}_lag1"]  = np.nan
                    cur[f"{c}_diff1"] = np.nan
            out_tbl = pa.Table.from_pandas(cur, preserve_index=False)
            pq.write_to_dataset(out_tbl, root_path=str(OUT_DIR), partition_cols=["safra"])
            prev       = cur[["msno"] + [c for c in BASE_COLS if c in cur.columns]].copy()
            prev_month = month
            continue

        is_consecutive = (add_1_month(prev_month) == month)

        if is_consecutive:
            cur = cur.merge(prev, on="msno", how="left", suffixes=("", "_prev"))
            for c in BASE_COLS:
                if c in cur.columns and f"{c}_prev" in cur.columns:
                    cur[f"{c}_lag1"]  = cur[f"{c}_prev"]
                    cur[f"{c}_diff1"] = cur[c] - cur[f"{c}_prev"]
                    cur.drop(columns=[f"{c}_prev"], inplace=True)
        else:
            for c in BASE_COLS:
                if c in cur.columns:
                    cur[f"{c}_lag1"]  = np.nan
                    cur[f"{c}_diff1"] = np.nan

        out_tbl = pa.Table.from_pandas(cur, preserve_index=False)
        pq.write_to_dataset(out_tbl, root_path=str(OUT_DIR), partition_cols=["safra"])

        prev       = cur[["msno"] + [c for c in BASE_COLS if c in cur.columns]].copy()
        prev_month = month

    print(f"OK → {OUT_DIR}")
    print("Obs: saída é um dataset parquet particionado por safra (pasta).")


if __name__ == "__main__":
    main()
