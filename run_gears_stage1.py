import os
import sys
import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# Keep environment setup before heavy imports
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class GearStage1Config:
    data_csv: str = str(PROJECT_ROOT / "data" / "train.csv")
    out_dir: str = str(PROJECT_ROOT / "gears_stage1_out")
    seed: int = 343
    test_frac: float = 0.2
    dev_frac: float = 0.2
    max_epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    hidden_size: int = 128
    n_hvg: int = 5000
    cell_col: str = "Cell_Line"
    drug_col: str = "Compound_LINCS_ID"
    split_key: str = "Compound_LINCS_ID"


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
    train = df[df[group_col].astype(str).isin(train_groups)].copy()
    dev = df[df[group_col].astype(str).isin(dev_groups)].copy()
    test = df[df[group_col].astype(str).isin(test_groups)].copy()
    return train, dev, test


def write_splits(train: pd.DataFrame, dev: pd.DataFrame, test: pd.DataFrame, out_dir: Path) -> Tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "xtrain_gears.csv"
    dev_path = out_dir / "xdev_gears.csv"
    test_path = out_dir / "xtest_gears.csv"
    train.to_csv(train_path, index=False)
    dev.to_csv(dev_path, index=False)
    test.to_csv(test_path, index=False)
    return train_path, dev_path, test_path


def main() -> None:
    cfg = GearStage1Config()
    set_seed(cfg.seed)

    data_csv = Path(cfg.data_csv)
    if not data_csv.exists():
        raise FileNotFoundError(f"Cannot find data csv: {data_csv}")

    df = pd.read_csv(data_csv)
    if cfg.cell_col not in df.columns:
        # 兼容只有转录组矩阵、没有显式 Cell_Line 的情况
        df[cfg.cell_col] = "UNKNOWN_CELL"
    if cfg.drug_col not in df.columns:
        # 兼容只有向量矩阵、没有显式 Compound_LINCS_ID 的情况
        df[cfg.drug_col] = [f"sample_{i}" for i in range(len(df))]

    train_df, dev_df, test_df = make_split(df, cfg.seed, cfg.split_key, cfg.dev_frac, cfg.test_frac)
    out_dir = Path(cfg.out_dir)
    train_path, dev_path, test_path = write_splits(train_df, dev_df, test_df, out_dir)

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
    (out_dir / "split_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("GEARS stage1 data prepared.")
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    print("\nSuggested next step:")
    print("- Use the generated CSVs with your installed GEARS API.")
    print("- If your GEARS package exposes PertData/GEARS, pass these files as the custom split.")
    print("- If your true GEARS input is a perturbation matrix, the script now falls back to synthetic `Cell_Line`/`Compound_LINCS_ID` columns so it can run without the original metadata.")


if __name__ == "__main__":
    main()
