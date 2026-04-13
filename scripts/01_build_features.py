from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def find_parquet(name: str) -> Path:
    p = RAW_DIR / f"{name}.parquet"
    if p.exists():
        return p
    # fallback: procura por padrão
    cands = sorted(RAW_DIR.rglob("*.parquet"))
    for x in cands:
        if name.lower() in x.name.lower():
            return x
    raise FileNotFoundError(f"Não encontrei parquet '{name}' em {RAW_DIR}")

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    df["msno"] = df["msno"].astype("string")
    df["safra"] = pd.to_numeric(df["safra"], errors="coerce").astype("int32")
    return df

def agg_add(base: pd.DataFrame | None, add: pd.DataFrame, keys=("msno","safra")) -> pd.DataFrame:
    """Soma incremental por chave (evita guardar lista de partes)."""
    if base is None:
        return add
    out = pd.concat([base, add], ignore_index=True)
    # soma apenas numéricos; para métricas de mean, tratamos separado
    return out.groupby(list(keys), as_index=False).sum(numeric_only=True)

def agg_user_logs(path: Path) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    cols = ["msno", "safra", "num_25","num_50","num_75","num_985","num_100","num_unq","total_secs"]
    cols = [c for c in cols if c in pf.schema.names]

    acc = None
    for rg in tqdm(range(pf.num_row_groups), desc="user_logs"):
        df = pf.read_row_group(rg, columns=cols).to_pandas()
        df = normalize(df)

        for c in ["num_25","num_50","num_75","num_985","num_100","num_unq","total_secs"]:
            if c not in df.columns:
                df[c] = 0
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        play_cols = ["num_25","num_50","num_75","num_985","num_100"]
        df["plays_total"] = df[play_cols].sum(axis=1)

        # vamos acumular numerador/denominador para calcular rates no final
        df["completion_num"] = (
            0.25*df["num_25"] + 0.50*df["num_50"] + 0.75*df["num_75"] + 0.985*df["num_985"] + 1.0*df["num_100"]
        )
        df["completion_den"] = df["plays_total"]
        df["secs_num"] = df["total_secs"]
        df["secs_den"] = df["plays_total"]

        g = df.groupby(["msno","safra"], as_index=False).agg(
            num_unq=("num_unq","sum"),
            total_secs=("total_secs","sum"),
            plays_total=("plays_total","sum"),
            num_25=("num_25","sum"),
            num_50=("num_50","sum"),
            num_75=("num_75","sum"),
            num_985=("num_985","sum"),
            num_100=("num_100","sum"),
            completion_num=("completion_num","sum"),
            completion_den=("completion_den","sum"),
            secs_num=("secs_num","sum"),
            secs_den=("secs_den","sum"),
        )

        acc = agg_add(acc, g)

    # rates finais
    acc["completion_rate"] = np.where(acc["completion_den"] > 0, acc["completion_num"] / acc["completion_den"], np.nan)
    acc["secs_per_play"] = np.where(acc["secs_den"] > 0, acc["secs_num"] / acc["secs_den"], np.nan)
    acc = acc.drop(columns=["completion_num","completion_den","secs_num","secs_den"], errors="ignore")
    return acc

def agg_transactions(path: Path) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    cols = ["msno","safra","actual_amount_paid","plan_list_price","payment_plan_days","is_auto_renew","is_cancel"]
    cols = [c for c in cols if c in pf.schema.names]

    # Para médias, acumulamos sum e count
    acc = None
    for rg in tqdm(range(pf.num_row_groups), desc="transactions"):
        df = pf.read_row_group(rg, columns=cols).to_pandas()
        df = normalize(df)

        for c in ["actual_amount_paid","plan_list_price","payment_plan_days","is_auto_renew","is_cancel"]:
            if c not in df.columns:
                df[c] = 0
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        df["plan_price_sum"] = df["plan_list_price"]
        df["plan_price_cnt"] = (df["plan_list_price"].notna()).astype("int32")
        df["plan_days_sum"] = df["payment_plan_days"]
        df["plan_days_cnt"] = (df["payment_plan_days"].notna()).astype("int32")
        df["auto_renew_sum"] = df["is_auto_renew"]
        df["auto_renew_cnt"] = (df["is_auto_renew"].notna()).astype("int32")

        g = df.groupby(["msno","safra"], as_index=False).agg(
            paid_sum=("actual_amount_paid","sum"),
            cancel_txn_count=("is_cancel","sum"),
            plan_price_sum=("plan_price_sum","sum"),
            plan_price_cnt=("plan_price_cnt","sum"),
            plan_days_sum=("plan_days_sum","sum"),
            plan_days_cnt=("plan_days_cnt","sum"),
            auto_renew_sum=("auto_renew_sum","sum"),
            auto_renew_cnt=("auto_renew_cnt","sum"),
        )

        acc = agg_add(acc, g)

    acc["plan_price_mean"] = np.where(acc["plan_price_cnt"] > 0, acc["plan_price_sum"] / acc["plan_price_cnt"], np.nan)
    acc["plan_days_mean"] = np.where(acc["plan_days_cnt"] > 0, acc["plan_days_sum"] / acc["plan_days_cnt"], np.nan)
    acc["auto_renew_rate"] = np.where(acc["auto_renew_cnt"] > 0, acc["auto_renew_sum"] / acc["auto_renew_cnt"], np.nan)

    acc = acc.drop(columns=[
        "plan_price_sum","plan_price_cnt","plan_days_sum","plan_days_cnt","auto_renew_sum","auto_renew_cnt"
    ], errors="ignore")
    return acc

def members_monthly(path: Path) -> pd.DataFrame:
    pf = pq.ParquetFile(path)
    cols = ["msno","safra","city","bd","gender","registered_via","is_ativo"]
    cols = [c for c in cols if c in pf.schema.names]

    # Como members é 2.8GB, ainda assim dá pra ler por row_group
    acc_parts = []
    for rg in tqdm(range(pf.num_row_groups), desc="members"):
        df = pf.read_row_group(rg, columns=cols).to_pandas()
        df = normalize(df)

        if "bd" in df.columns:
            df["bd"] = pd.to_numeric(df["bd"], errors="coerce")
            df.loc[(df["bd"] < 10) | (df["bd"] > 90), "bd"] = np.nan

        if "is_ativo" in df.columns:
            df["is_ativo"] = pd.to_numeric(df["is_ativo"], errors="coerce")

        acc_parts.append(df)

    out = pd.concat(acc_parts, ignore_index=True)

    agg = {}
    for c in ["city","bd","gender","registered_via"]:
        if c in out.columns:
            agg[c] = "first"
    if "is_ativo" in out.columns:
        agg["is_ativo"] = "max"

    out = out.groupby(["msno","safra"], as_index=False).agg(agg)
    return out

def main():
    ul_path = find_parquet("user_logs")
    tx_path = find_parquet("transactions")
    mb_path = find_parquet("members")

    print("Usando:")
    print(" -", ul_path)
    print(" -", tx_path)
    print(" -", mb_path)

    ul_feat = agg_user_logs(ul_path)
    tx_feat = agg_transactions(tx_path)
    mb_feat = members_monthly(mb_path)

    feat = ul_feat.merge(tx_feat, on=["msno","safra"], how="left").merge(mb_feat, on=["msno","safra"], how="left")
    feat["has_payment"] = (feat["paid_sum"].fillna(0) > 0).astype("int8")
    feat["has_usage"] = (feat["total_secs"].fillna(0) > 0).astype("int8")

    feat.to_parquet(OUT_PATH, index=False)
    print("OK ->", OUT_PATH, "| rows:", len(feat), "| cols:", feat.shape[1])

if __name__ == "__main__":
    main()