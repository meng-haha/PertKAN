"""
加载已保存的 best_toxicity_kan.pth，在指定数据上跑毒性分类评估（ACC、Macro-F1、classification report）。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score

from models import DeepCE
from models.toxicity_classifier import DeepCETwitteroxicityKAN, ToxicityHeadConfig
from utils.data_utils import read_cell, read_gene
from utils.toxicity_datareader import ToxLabelSpec, ToxicityCSVDataset, create_dataloader, split_indices_stratified


def load_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" not in ckpt:
        raise ValueError("checkpoint 格式不正确：缺少 model_state_dict")
    return ckpt


def main():
    parser = argparse.ArgumentParser(description="加载已保存的毒性模型，在 CSV 上跑分类评估")
    parser.add_argument("--ckpt", required=True, help="main_toxicity.py 保存的 best_toxicity_kan.pth")
    parser.add_argument("--tox_csv", default="data/log_normalized_data.csv", help="毒性 CSV（与训练格式一致）")
    parser.add_argument("--split", choices=["full", "test"], default="test",
                        help="full=整表评估，test=仅用与训练时相同划分的测试集")
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=343)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_dir", default=None, help="若指定，将预测结果和指标写入该目录")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.ckpt, device=device)

    # 从 ckpt 恢复 label_spec（需与训练时一致）
    ls = ckpt.get("label_spec", {})
    if isinstance(ls, dict):
        label_spec = ToxLabelSpec(
            threshold_non_toxic=float(ls.get("threshold_non_toxic", 0.5)),
            label_names=tuple(ls.get("label_names", ["cardiac", "hepatic", "nervous", "renal", "non_toxic"])),
        )
    else:
        label_spec = ToxLabelSpec()

    full_ds = ToxicityCSVDataset(args.tox_csv, label_spec=label_spec)
    label_names = list(label_spec.label_names)

    if args.split == "test":
        train_idx, dev_idx, test_idx = split_indices_stratified(
            [full_ds[i]["label"] for i in range(len(full_ds))],
            dev_ratio=float(args.dev_ratio),
            test_ratio=float(args.test_ratio),
            seed=int(args.seed),
        )
        eval_ds = torch.utils.data.Subset(full_ds, test_idx.tolist())
        print(f"使用测试集样本数: {len(eval_ds)}")
    else:
        eval_ds = full_ds
        print(f"使用全量样本数: {len(eval_ds)}")

    gene_file = ckpt.get("gene_file", "data/DeepCE_gene_vectors.csv")
    gene = read_gene(gene_file, device=device).float()

    cell_features = None
    default_cell_idx = 0
    include_cell_id = False
    if ckpt.get("cell_file") is not None:
        cell_features, cell_id_to_idx = read_cell(ckpt["cell_file"], device=device)
        include_cell_id = True
        default_cell_id = ckpt.get("default_cell_id")
        if default_cell_id is not None and default_cell_id in cell_id_to_idx:
            default_cell_idx = int(cell_id_to_idx[default_cell_id])

    loader = create_dataloader(
        eval_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        device=device,
        include_cell_id=include_cell_id,
        default_cell_idx=default_cell_idx,
    )

    # 构建模型（与 predict_toxicity.py 一致）
    deepce_ckpt = ckpt.get("deepce_ckpt")
    head_cfg_dict = ckpt.get("head_cfg", {})
    deepce = DeepCE(
        drug_input_dim={"atom": 62, "bond": 6},
        drug_emb_dim=128,
        conv_size=[16, 16],
        degree=[0, 1, 2, 3, 4, 5],
        gene_input_dim=int(gene.shape[1]),
        gene_emb_dim=128,
        num_gene=int(gene.shape[0]),
        hid_dim=128,
        dropout=float(head_cfg_dict.get("dropout", 0.1)),
        loss_type="cross_entropy",
        device=device,
        initializer=torch.nn.init.xavier_uniform_,
        cell_id_input_dim=int(cell_features.shape[1]) if cell_features is not None else 0,
        cell_id_emb_dim=4,
        use_cell_id=include_cell_id,
    ).float().to(device)
    if deepce_ckpt is not None:
        state = torch.load(deepce_ckpt, map_location=device)
        if isinstance(state, dict):
            state = state.get("model_state_dict") or state.get("state_dict") or state
        deepce.load_state_dict(state, strict=True)

    head_cfg = ToxicityHeadConfig(**head_cfg_dict) if head_cfg_dict else ToxicityHeadConfig()
    model = DeepCETwitteroxicityKAN(deepce=deepce, head_cfg=head_cfg, freeze_deepce=True).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    ys, ps = [], []
    with torch.no_grad():
        for ft, y in loader:
            logits = model(
                drug=ft["drug"],
                gene=gene,
                mask=ft["mask"],
                cell_id=ft.get("cell_id"),
                cell_features=cell_features,
            )
            pred = torch.argmax(logits, dim=-1)
            ys.append(y.detach().cpu().numpy())
            ps.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(ps, axis=0)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    rep = classification_report(y_true, y_pred, target_names=label_names, digits=4)

    print("\n===== 毒性分类评估 =====")
    print(f"ACC = {acc:.4f} | Macro-F1 = {f1:.4f}")
    print(rep)

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics = {"accuracy": float(acc), "macro_f1": float(f1), "classification_report": rep}
        (out_dir / "eval_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        np.savez(out_dir / "eval_preds.npz", y_true=y_true, y_pred=y_pred)
        print(f"已保存指标与预测: {out_dir}")


if __name__ == "__main__":
    main()
