"""
根据 dev_predictions.csv 自动搜索每个毒性任务的最佳分类阈值，
并用这些阈值重新评估 dev/test_predictions.csv。

默认适配 toxicity_stage2/toxicity_stage2_out 目录下的预测文件。

示例：
    python tune_toxicity_thresholds.py

    python tune_toxicity_thresholds.py \
        --pred_dir toxicity_stage2/toxicity_stage2_out \
        --metric f1 \
        --min_threshold 0.05 \
        --max_threshold 0.95 \
        --step 0.01
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


DEFAULT_TASKS = ["cardiac", "hepatic", "nervous", "renal"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在 dev 集上搜索每个毒性任务的最佳阈值，并重新评估 dev/test 集。"
    )
    parser.add_argument(
        "--pred_dir",
        default="toxicity_stage2/toxicity_stage2_out",
        help="包含 dev_predictions.csv 和 test_predictions.csv 的目录。",
    )
    parser.add_argument(
        "--dev_file",
        default=None,
        help="dev 预测 CSV 路径；不填则使用 pred_dir/dev_predictions.csv。",
    )
    parser.add_argument(
        "--test_file",
        default=None,
        help="test 预测 CSV 路径；不填则使用 pred_dir/test_predictions.csv。",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=DEFAULT_TASKS,
        help="要调阈值的任务名，对应 {task}_true_cls 和 {task}_pred 列。",
    )
    parser.add_argument(
        "--metric",
        choices=["f1", "balanced_accuracy", "accuracy", "youden", "precision_at_recall"],
        default="f1",
        help="dev 集阈值选择指标。youden = recall + specificity - 1。",
    )
    parser.add_argument(
        "--target_recall",
        type=float,
        default=0.70,
        help="metric=precision_at_recall 时，要求 recall 至少达到该值。",
    )
    parser.add_argument("--min_threshold", type=float, default=0.05, help="搜索阈值下限。")
    parser.add_argument("--max_threshold", type=float, default=0.95, help="搜索阈值上限。")
    parser.add_argument("--step", type=float, default=0.01, help="阈值搜索步长。")
    parser.add_argument("--out_dir", default=None, help="输出目录；不填则输出到 pred_dir/threshold_tuning。")
    parser.add_argument(
        "--save_adjusted_predictions",
        action="store_true",
        help="是否额外保存使用最佳阈值后的 dev/test 预测文件。",
    )
    return parser.parse_args()


def make_threshold_grid(min_threshold: float, max_threshold: float, step: float) -> np.ndarray:
    if not 0.0 <= min_threshold <= 1.0:
        raise ValueError("min_threshold 必须在 [0, 1] 范围内。")
    if not 0.0 <= max_threshold <= 1.0:
        raise ValueError("max_threshold 必须在 [0, 1] 范围内。")
    if min_threshold > max_threshold:
        raise ValueError("min_threshold 不能大于 max_threshold。")
    if step <= 0:
        raise ValueError("step 必须大于 0。")

    count = int(round((max_threshold - min_threshold) / step)) + 1
    grid = min_threshold + np.arange(count) * step
    grid = np.clip(grid, min_threshold, max_threshold)
    return np.unique(np.round(grid, 8))


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": safe_auc(y_true, y_score),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "positive_rate_pred": float(y_pred.mean()),
        "positive_rate_true": float(y_true.mean()),
    }


def threshold_score(metrics: Dict[str, float], metric_name: str, target_recall: float) -> float:
    if metric_name == "youden":
        return metrics["recall"] + metrics["specificity"] - 1.0
    if metric_name == "precision_at_recall":
        if metrics["recall"] < target_recall:
            return -1.0
        return metrics["precision"]
    return metrics[metric_name]


def find_best_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: Iterable[float],
    metric_name: str,
    target_recall: float,
) -> Tuple[float, Dict[str, float], List[Dict[str, float]]]:
    rows: List[Dict[str, float]] = []
    best_threshold = 0.5
    best_metrics: Dict[str, float] = {}
    best_score = -np.inf

    for threshold in thresholds:
        metrics = binary_metrics(y_true, y_score, float(threshold))
        score = threshold_score(metrics, metric_name, target_recall)
        metrics["selection_score"] = float(score)
        rows.append(metrics)

        # 分数相同时，优先选择更接近 0.5 的阈值，避免过于极端。
        better_score = score > best_score
        same_score_better_threshold = np.isclose(score, best_score) and abs(float(threshold) - 0.5) < abs(best_threshold - 0.5)
        if better_score or same_score_better_threshold:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics, rows


def require_columns(df: pd.DataFrame, tasks: Iterable[str]) -> None:
    missing = []
    for task in tasks:
        for suffix in ["true_cls", "pred"]:
            col = f"{task}_{suffix}"
            if col not in df.columns:
                missing.append(col)
    if missing:
        raise ValueError("预测文件缺少必要列：" + ", ".join(missing))


def tune_thresholds(
    dev_df: pd.DataFrame,
    tasks: List[str],
    thresholds: np.ndarray,
    metric_name: str,
    target_recall: float,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame]:
    best_thresholds: Dict[str, float] = {}
    best_rows = []
    curve_rows = []

    for task in tasks:
        y_true = dev_df[f"{task}_true_cls"].to_numpy(dtype=int)
        y_score = dev_df[f"{task}_pred"].to_numpy(dtype=float)
        best_threshold, best_metrics, task_curve = find_best_threshold(
            y_true=y_true,
            y_score=y_score,
            thresholds=thresholds,
            metric_name=metric_name,
            target_recall=target_recall,
        )
        best_thresholds[task] = best_threshold
        best_rows.append({"task": task, **best_metrics})
        for row in task_curve:
            curve_rows.append({"task": task, **row})

    return best_thresholds, pd.DataFrame(best_rows), pd.DataFrame(curve_rows)


def evaluate_with_thresholds(df: pd.DataFrame, tasks: List[str], thresholds: Dict[str, float], split_name: str) -> pd.DataFrame:
    rows = []
    for task in tasks:
        y_true = df[f"{task}_true_cls"].to_numpy(dtype=int)
        y_score = df[f"{task}_pred"].to_numpy(dtype=float)
        row = binary_metrics(y_true, y_score, thresholds[task])
        rows.append({"split": split_name, "task": task, **row})
    return pd.DataFrame(rows)


def summarize_eval(eval_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for split, part in eval_df.groupby("split"):
        summary[str(split)] = {
            "avg_accuracy": float(part["accuracy"].mean()),
            "avg_balanced_accuracy": float(part["balanced_accuracy"].mean()),
            "avg_precision": float(part["precision"].mean()),
            "avg_recall": float(part["recall"].mean()),
            "avg_specificity": float(part["specificity"].mean()),
            "avg_f1": float(part["f1"].mean()),
            "auc_macro": float(part["auc"].mean()),
        }
    return summary


def add_adjusted_prediction_columns(df: pd.DataFrame, tasks: List[str], thresholds: Dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    for task in tasks:
        out[f"{task}_threshold_tuned"] = thresholds[task]
        out[f"{task}_pred_cls_tuned"] = (out[f"{task}_pred"].astype(float) >= thresholds[task]).astype(int)
    return out


def main() -> None:
    args = parse_args()
    pred_dir = Path(args.pred_dir)
    dev_file = Path(args.dev_file) if args.dev_file else pred_dir / "dev_predictions.csv"
    test_file = Path(args.test_file) if args.test_file else pred_dir / "test_predictions.csv"
    out_dir = Path(args.out_dir) if args.out_dir else pred_dir / "threshold_tuning"
    out_dir.mkdir(parents=True, exist_ok=True)

    dev_df = pd.read_csv(dev_file)
    test_df = pd.read_csv(test_file)
    tasks = list(args.tasks)
    require_columns(dev_df, tasks)
    require_columns(test_df, tasks)

    thresholds = make_threshold_grid(args.min_threshold, args.max_threshold, args.step)
    best_thresholds, best_dev_df, threshold_curve_df = tune_thresholds(
        dev_df=dev_df,
        tasks=tasks,
        thresholds=thresholds,
        metric_name=args.metric,
        target_recall=args.target_recall,
    )

    eval_df = pd.concat(
        [
            evaluate_with_thresholds(dev_df, tasks, best_thresholds, "dev"),
            evaluate_with_thresholds(test_df, tasks, best_thresholds, "test"),
        ],
        ignore_index=True,
    )
    summary = summarize_eval(eval_df)

    best_dev_df.to_csv(out_dir / "best_thresholds_dev.csv", index=False, encoding="utf-8-sig")
    threshold_curve_df.to_csv(out_dir / "threshold_search_curve.csv", index=False, encoding="utf-8-sig")
    eval_df.to_csv(out_dir / "metrics_with_tuned_thresholds.csv", index=False, encoding="utf-8-sig")

    result = {
        "dev_file": str(dev_file),
        "test_file": str(test_file),
        "selection_metric": args.metric,
        "target_recall": float(args.target_recall),
        "threshold_grid": {
            "min": float(args.min_threshold),
            "max": float(args.max_threshold),
            "step": float(args.step),
        },
        "best_thresholds": best_thresholds,
        "summary": summary,
    }
    (out_dir / "threshold_tuning_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.save_adjusted_predictions:
        add_adjusted_prediction_columns(dev_df, tasks, best_thresholds).to_csv(
            out_dir / "dev_predictions_threshold_tuned.csv", index=False, encoding="utf-8-sig"
        )
        add_adjusted_prediction_columns(test_df, tasks, best_thresholds).to_csv(
            out_dir / "test_predictions_threshold_tuned.csv", index=False, encoding="utf-8-sig"
        )

    print("\n===== 阈值调优完成 =====")
    print(f"输出目录: {out_dir}")
    print("\n最佳阈值（来自 dev 集）:")
    for task, threshold in best_thresholds.items():
        print(f"  {task}: {threshold:.4f}")

    print("\n使用调优阈值后的平均指标:")
    for split, metrics in summary.items():
        print(
            f"  {split}: "
            f"F1={metrics['avg_f1']:.4f}, "
            f"Precision={metrics['avg_precision']:.4f}, "
            f"Recall={metrics['avg_recall']:.4f}, "
            f"BalancedACC={metrics['avg_balanced_accuracy']:.4f}, "
            f"AUC={metrics['auc_macro']:.4f}"
        )


if __name__ == "__main__":
    main()
