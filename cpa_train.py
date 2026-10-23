import argparse
from pathlib import Path

import pandas as pd


try:
    from cpa import CPA
except Exception as exc:  # pragma: no cover
    CPA = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def main():
    parser = argparse.ArgumentParser(description="Train CPA on stage-1 transcriptome data")
    parser.add_argument("--data_csv", required=True)
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--dev_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if CPA is None:
        raise ImportError(
            "CPA package is not installed or could not be imported. "
            f"Original error: {_IMPORT_ERROR}"
        )

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    dev_df = pd.read_csv(args.dev_csv)
    test_df = pd.read_csv(args.test_csv)

    model = CPA(seed=args.seed)
    model.train(train_df=train_df, dev_df=dev_df, test_df=test_df, save_dir=str(save_dir))
    print(f"CPA training finished. Outputs saved to {save_dir}")


if __name__ == "__main__":
    main()
