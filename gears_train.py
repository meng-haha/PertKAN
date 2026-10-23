import argparse
from pathlib import Path

import pandas as pd


try:
    from gears import GEARS
except Exception as exc:  # pragma: no cover
    GEARS = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def main():
    parser = argparse.ArgumentParser(description="Train GEARS on stage-1 transcriptome data")
    parser.add_argument("--data_csv", required=True)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--dev_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if GEARS is None:
        raise ImportError(
            "GEARS package is not installed or could not be imported. "
            f"Original error: {_IMPORT_ERROR}"
        )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    dev_df = pd.read_csv(args.dev_csv)
    test_df = pd.read_csv(args.test_csv)

    # 这里保持最小假设：你的数据已经是表达矩阵格式，可直接交给 GEARS 训练接口
    model = GEARS(seed=args.seed)
    model.train(train_df=train_df, dev_df=dev_df, test_df=test_df, save_dir=str(save_dir))
    print(f"GEARS training finished. Outputs saved to {save_dir}")


if __name__ == "__main__":
    main()
