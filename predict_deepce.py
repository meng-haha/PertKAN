import os
# 避免 libgomp: Invalid value for environment variable OMP_NUM_THREADS（必须在 import numpy/torch 之前）
try:
    n = (os.environ.get("OMP_NUM_THREADS") or "").strip()
    if not n or n == "0" or (not n.isdigit()) or int(n) < 1:
        os.environ["OMP_NUM_THREADS"] = "1"
except Exception:
    os.environ["OMP_NUM_THREADS"] = "1"

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from models import DeepCE
from utils import DataReader, classification_metrics


@torch.inference_mode()
def eval_phase(model, data, batch_size: int, phase: str):
    # 模型为 float32，DataReader 的 gene/cell 为 float64，需转 float32 避免 mat1/mat2 dtype 报错
    gene = data.gene.float()
    cell = data.cell.float() if data.cell is not None else None
    all_labels, all_preds, all_probs = [], [], []
    for ft, lb in data.get_batch_data(dataset=phase, batch_size=batch_size, shuffle=False):
        ft["drug"]["atom"] = ft["drug"]["atom"].float()
        ft["drug"]["bond"] = ft["drug"]["bond"].float()
        pred = model(ft["drug"], gene, ft["mask"], ft.get("cell_id"), cell)
        pred_cls = torch.argmax(pred, dim=-1)
        pred_prob = torch.softmax(pred, dim=-1)
        all_labels.append(lb.cpu().numpy())
        all_preds.append(pred_cls.cpu().numpy())
        all_probs.append(pred_prob.cpu().numpy())

    y_t = np.concatenate(all_labels, axis=0).ravel()
    y_p = np.concatenate(all_preds, axis=0).ravel()
    y_prob = np.concatenate(all_probs, axis=0).reshape(-1, 3)

    acc, f1, report = classification_metrics(y_t, y_p)
    macro_auc = roc_auc_score(y_t, y_prob, multi_class="ovr", average="macro")
    return acc, f1, macro_auc, report


def main():
    parser = argparse.ArgumentParser(description="DeepCE checkpoint 推理检查（dev/test 指标）")
    parser.add_argument("--ckpt", required=True, help="main_deepce.py 保存的 deepce_best.pth 或 deepce_last.pth")
    parser.add_argument("--drug_file", default="data/drug_smiles.txt")
    parser.add_argument("--gene_file", default="data/DeepCE_gene_vectors.csv")
    parser.add_argument("--train_file", default="data/train.csv")
    parser.add_argument("--dev_file", default="data/dev.csv")
    parser.add_argument("--test_file", default="data/test.csv")
    parser.add_argument("--cell_file", default="data/cell_features.csv")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="基因数大时 attention 显存约 O(num_gene^2)，OOM 请保持 1；显存够可试 2/4/8")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = DataReader(
        args.drug_file,
        args.gene_file,
        args.train_file,
        args.dev_file,
        args.test_file,
        filter={},
        device=device,
        cell_file=args.cell_file,
    )

    model = DeepCE(
        drug_input_dim={"atom": 62, "bond": 6},
        drug_emb_dim=128,
        conv_size=[16, 16],
        degree=[0, 1, 2, 3, 4, 5],
        gene_input_dim=int(np.shape(data.gene)[1]),
        gene_emb_dim=128,
        num_gene=int(np.shape(data.gene)[0]),
        hid_dim=128,
        dropout=0.0,
        loss_type="cross_entropy",
        device=device,
        initializer=torch.nn.init.xavier_uniform_,
        cell_id_input_dim=int(np.shape(data.cell)[1]) if data.cell is not None else 0,
        cell_id_emb_dim=4,
        use_cell_id=data.use_cell_id,
    ).float().to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    for phase in ["dev", "test"]:
        acc, f1, macro_auc, report = eval_phase(model, data, batch_size=int(args.batch_size), phase=phase)
        print(f"\n===== {phase.upper()} =====")
        print(f"ACC={acc:.4f} | Macro-F1={f1:.4f} | Macro-AUC={macro_auc:.4f}")
        print(report)


if __name__ == "__main__":
    main()

