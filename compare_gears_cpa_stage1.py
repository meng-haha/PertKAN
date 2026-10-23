import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


DEFAULT_DATA_CSV = Path(r"f:\Toxic\DeepCR\DeepCE-master\DeepCE-master\DeepCE\MyNewData\MyNewCode\data\xtrain.csv")
DEFAULT_OUT_DIR = Path(r"f:\Toxic\DeepCR\DeepCE-master\DeepCE-master\DeepCE\MyNewData\MyNewCode\stage1_compare_out")
SMILES_COL_CANDIDATES = ["queried_smiles", "smiles", "SMILES"]
CELL_COL_CANDIDATES = ["Cell_Line", "cell_id", "cell", "cell_line"]
PERT_COL_CANDIDATES = ["Compound_LINCS_ID", "pert_id", "drug", "compound"]


def resolve_col(columns: List[str], candidates: List[str], required: bool = True) -> str:
    for c in candidates:
        if c in columns:
            return c
    if required:
        raise ValueError(f"Cannot find any of columns {candidates} in input file.")
    return ""


def split_by_pert(df: pd.DataFrame, seed: int = 42, train_ratio: float = 0.7, dev_ratio: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    pert_ids = df["_pert_key"].dropna().unique().tolist()
    rng.shuffle(pert_ids)
    n = len(pert_ids)
    n_train = max(1, int(n * train_ratio))
    n_dev = max(1, int(n * dev_ratio))
    train_ids = set(pert_ids[:n_train])
    dev_ids = set(pert_ids[n_train:n_train + n_dev])
    test_ids = set(pert_ids[n_train + n_dev:])
    train_df = df[df["_pert_key"].isin(train_ids)].copy()
    dev_df = df[df["_pert_key"].isin(dev_ids)].copy()
    test_df = df[df["_pert_key"].isin(test_ids)].copy()
    return train_df, dev_df, test_df


def save_splits(train_df: pd.DataFrame, dev_df: pd.DataFrame, test_df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.drop(columns=["_pert_key"]).to_csv(out_dir / "train.csv", index=False)
    dev_df.drop(columns=["_pert_key"]).to_csv(out_dir / "dev.csv", index=False)
    test_df.drop(columns=["_pert_key"]).to_csv(out_dir / "test.csv", index=False)


def build_gears_command(out_dir: Path, data_csv: Path, seed: int):
    return (
        "python gears_train.py "
        f"--data_csv \"{data_csv}\" "
        f"--train_csv \"{out_dir / 'train.csv'}\" "
        f"--dev_csv \"{out_dir / 'dev.csv'}\" "
        f"--test_csv \"{out_dir / 'test.csv'}\" "
        f"--save_dir \"{out_dir / 'gears_out'}\" "
        f"--seed {seed}"
    )


def build_cpa_command(out_dir: Path, data_csv: Path, seed: int):
    return (
        "python cpa_train.py "
        f"--data_csv \"{data_csv}\" "
        f"--train_csv \"{out_dir / 'train.csv'}\" "
        f"--dev_csv \"{out_dir / 'dev.csv'}\" "
        f"--test_csv \"{out_dir / 'test.csv'}\" "
        f"--save_dir \"{out_dir / 'cpa_out'}\" "
        f"--seed {seed}"
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare stage-1 transcriptome comparison for GEARS and CPA")
    parser.add_argument("--data_csv", type=str, default=str(DEFAULT_DATA_CSV))
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", choices=["gears", "cpa", "all"], default="all")
    args = parser.parse_args()

    data_csv = Path(args.data_csv)
    out_dir = Path(args.output_dir)
    df = pd.read_csv(data_csv)

    cell_col = resolve_col(list(df.columns), CELL_COL_CANDIDATES)
    pert_col = resolve_col(list(df.columns), PERT_COL_CANDIDATES)
    smiles_col = resolve_col(list(df.columns), SMILES_COL_CANDIDATES, required=False)

    df = df.copy()
    if smiles_col:
        df = df.rename(columns={smiles_col: "smiles"})
    df["_pert_key"] = df[pert_col].astype(str)
    df["_cell_key"] = df[cell_col].astype(str)

    train_df, dev_df, test_df = split_by_pert(df, seed=args.seed)
    save_splits(train_df, dev_df, test_df, out_dir)

    meta = {
        "data_csv": str(data_csv),
        "output_dir": str(out_dir),
        "seed": args.seed,
        "num_rows": int(len(df)),
        "num_train": int(len(train_df)),
        "num_dev": int(len(dev_df)),
        "num_test": int(len(test_df)),
        "cell_col": cell_col,
        "pert_col": pert_col,
        "smiles_col": smiles_col,
    }
    pd.Series(meta).to_csv(out_dir / "split_meta.csv")

    print("Prepared splits:")
    for k, v in meta.items():
        print(f"  {k}: {v}")

    if args.method in ("gears", "all"):
        print("\nGEARS command:")
        print(build_gears_command(out_dir, data_csv, args.seed))
    if args.method in ("cpa", "all"):
        print("\nCPA command:")
        print(build_cpa_command(out_dir, data_csv, args.seed))


if __name__ == "__main__":
    main()
