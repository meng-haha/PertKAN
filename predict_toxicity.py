import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import torch

from models import DeepCE
from models.toxicity_classifier import DeepCETwitteroxicityKAN, ToxicityHeadConfig
from utils.data_utils import convert_smile_to_feature, create_mask_feature, read_cell, read_gene


def load_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" not in ckpt:
        raise ValueError("checkpoint 格式不正确：缺少 model_state_dict")
    return ckpt


@torch.inference_mode()
def predict_batch(
    model: DeepCETwitteroxicityKAN,
    smiles_list: List[str],
    gene: torch.Tensor,
    device: torch.device,
    label_names: List[str],
    cell_features: Optional[torch.Tensor] = None,
    default_cell_idx: int = 0,
):
    drug = convert_smile_to_feature(smiles_list, device=device)
    mask = create_mask_feature(drug, device=device)
    cell_id = None
    if cell_features is not None:
        cell_id = torch.full((len(smiles_list),), int(default_cell_idx), dtype=torch.long, device=device)

    proba = model.predict_proba(drug=drug, gene=gene, mask=mask, cell_id=cell_id, cell_features=cell_features)
    top = torch.argmax(proba, dim=-1).detach().cpu().tolist()
    proba = proba.detach().cpu().tolist()
    results = []
    for s, t, p in zip(smiles_list, top, proba):
        results.append(
            {
                "smiles": s,
                "pred_class": int(t),
                "pred_name": label_names[int(t)],
                "proba": {label_names[i]: float(p[i]) for i in range(len(label_names))},
            }
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="KAN 毒性 5 分类推理（基于 DeepKAN 输出）")
    parser.add_argument("--ckpt", required=True, help="main_toxicity.py 保存的 best_toxicity_kan.pth")
    parser.add_argument("--smiles", default=None, help="单个 SMILES 字符串")
    parser.add_argument("--smiles_file", default=None, help="每行一个 SMILES 的文本文件")
    parser.add_argument("--out_json", default="tox_pred.json")
    parser.add_argument("--device", default=None, help="cuda / cpu")
    args = parser.parse_args()

    if args.smiles is None and args.smiles_file is None:
        raise ValueError("请提供 --smiles 或 --smiles_file")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(args.ckpt, device=device)

    label_spec = ckpt.get("label_spec", {})
    label_names = label_spec.get("label_names", ["cardiac", "hepatic", "nervous", "renal", "non_toxic"])

    gene_file = ckpt.get("gene_file", "data/DeepCE_gene_vectors.csv")
    gene = read_gene(gene_file, device=device).float()

    cell_features = None
    default_cell_idx = 0
    if ckpt.get("cell_file") is not None:
        cell_features, cell_id_to_idx = read_cell(ckpt["cell_file"], device=device)
        default_cell_id = ckpt.get("default_cell_id")
        if default_cell_id is not None and default_cell_id in cell_id_to_idx:
            default_cell_idx = int(cell_id_to_idx[default_cell_id])

    # DeepCE base（结构必须与训练时一致）
    deepce_ckpt = ckpt.get("deepce_ckpt")
    deepce = DeepCE(
        drug_input_dim={"atom": 62, "bond": 6},
        drug_emb_dim=128,
        conv_size=[16, 16],
        degree=[0, 1, 2, 3, 4, 5],
        gene_input_dim=int(gene.shape[1]),
        gene_emb_dim=128,
        num_gene=int(gene.shape[0]),
        hid_dim=128,
        dropout=float(ckpt.get("head_cfg", {}).get("dropout", 0.1)),
        loss_type="cross_entropy",
        device=device,
        initializer=torch.nn.init.xavier_uniform_,
        cell_id_input_dim=int(cell_features.shape[1]) if cell_features is not None else 0,
        cell_id_emb_dim=4,
        use_cell_id=cell_features is not None,
    ).float().to(device)
    if deepce_ckpt is not None:
        state = torch.load(deepce_ckpt, map_location=device)
        if isinstance(state, dict):
            state = state.get("model_state_dict") or state.get("state_dict") or state
        deepce.load_state_dict(state, strict=True)

    head_cfg = ToxicityHeadConfig(**ckpt.get("head_cfg", asdict(ToxicityHeadConfig())))
    model = DeepCETwitteroxicityKAN(deepce=deepce, head_cfg=head_cfg, freeze_deepce=True).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    smiles_list: List[str]
    if args.smiles_file is not None:
        smiles_list = [line.strip() for line in Path(args.smiles_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        smiles_list = [args.smiles.strip()]

    results = predict_batch(
        model=model,
        smiles_list=smiles_list,
        gene=gene,
        device=device,
        label_names=list(label_names),
        cell_features=cell_features,
        default_cell_idx=default_cell_idx,
    )
    Path(args.out_json).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存: {args.out_json}")


if __name__ == "__main__":
    main()
#单条推理
# python predict_toxicity.py --ckpt tox_out/best_toxicity_kan.pth --smiles "CC(=O)NC1=CC=C(C=C1)O" --out_json pred.json
# 批量推理
#python predict_toxicity.py --ckpt tox_out/best_toxicity_kan.pth --smiles_file data/drug_smiles.txt --out_json pred.json