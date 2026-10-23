import os
import sys
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class CPAStage1Config:
    data_csv: str = str(PROJECT_ROOT / "data" / "log_normalized_data_plus_tox21neg.csv")
    out_dir: str = str(PROJECT_ROOT / "cpa_stage1_out")
    seed: int = 343
    dev_frac: float = 0.2
    test_frac: float = 0.2
    cell_col: str = "Cell_Line"
    drug_col: str = "Compound_LINCS_ID"
    dose_col: str = "pert_idose"
    batch_col: str = "batch"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def make_split(df: pd.DataFrame, seed: int, group_col: str, dev_frac: float, test_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = df[group_col].dropna().astype(str).unique().tolist()
    rng = np.random.RandomState(seed)
    rng.shuffle(groups)
    n = len(groups)
    n_test = max(1, int(round(n * test_frac)))
    n_dev = max(1, int(round(n * dev_frac)))
    test_groups = set(groups[:n_test])
    dev_groups = set(groups[n_test:n_test + n_dev])
    train_groups = set(groups[n_test + n_dev:])
    return (
        df[df[group_col].astype(str).isin(train_groups)].copy(),
        df[df[group_col].astype(str).isin(dev_groups)].copy(),
        df[df[group_col].astype(str).isin(test_groups)].copy(),
    )


def build_cpa_table(df: pd.DataFrame, out_path: Path, split_name: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = df.copy()
    tmp["split"] = split_name
    if "dose_val" not in tmp.columns:
        tmp["dose_val"] = 1.0
    if "condition" not in tmp.columns:
        tmp["condition"] = tmp["Compound_LINCS_ID"].astype(str)
    if "cell_type" not in tmp.columns:
        tmp["cell_type"] = tmp["Cell_Line"].astype(str)
    tmp.to_csv(out_path, index=False)


def main() -> None:
    cfg = CPAStage1Config()
    set_seed(cfg.seed)

    data_csv = Path(cfg.data_csv)
    if not data_csv.exists():
        raise FileNotFoundError(f"Cannot find data csv: {data_csv}")

    df = pd.read_csv(data_csv)
    for col in [cfg.cell_col, cfg.drug_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    train_df, dev_df, test_df = make_split(df, cfg.seed, cfg.cell_col, cfg.dev_frac, cfg.test_frac)
    out_dir = Path(cfg.out_dir)
    train_path = out_dir / "train_cpa.csv"
    dev_path = out_dir / "dev_cpa.csv"
    test_path = out_dir / "test_cpa.csv"
    build_cpa_table(train_df, train_path, "train")
    build_cpa_table(dev_df, dev_path, "dev")
    build_cpa_table(test_df, test_path, "test")

    meta = {
        "config": asdict(cfg),
        "n_total": int(len(df)),
        "n_train": int(len(train_df)),
        "n_dev": int(len(dev_df)),
        "n_test": int(len(test_df)),
        "train_file": str(train_path),
        "dev_file": str(dev_path),
        "test_file": str(test_path),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "split_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("CPA stage1 data prepared.")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print("\nSuggested next step:")
    print("- Feed the generated CSVs into your CPA training pipeline.")
    print("- If your CPA implementation expects a specific AnnData or .h5ad format, I can add a converter next.")


if __name__ == "__main__":
    main()
