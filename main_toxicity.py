import os
# 最先设置，避免 libgomp: Invalid value for environment variable OMP_NUM_THREADS
try:
    n = os.environ.get("OMP_NUM_THREADS", "1").strip()
    if not n or n == "0" or not n.isdigit():
        os.environ["OMP_NUM_THREADS"] = "1"
except Exception:
    os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score

from models import DeepCE
from models.toxicity_classifier import DeepCETwitteroxicityKAN, ToxicityHeadConfig
from utils.data_utils import read_cell, read_gene
from utils.toxicity_datareader import ToxLabelSpec, ToxicityCSVDataset, create_dataloader, split_indices_stratified


def set_seed(seed: int = 343):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="KAN 毒性 5 分类训练（基于 DeepKAN 输出）")
    parser.add_argument("--tox_csv", default="data/log_normalized_data.csv")
    parser.add_argument("--gene_file", default="data/DeepCE_gene_vectors.csv")
    parser.add_argument("--cell_file", default=None)
    parser.add_argument("--default_cell_id", default=None, help="当 DeepKAN 需要 cell_id 时使用的 cell 名称")
    parser.add_argument("--deepce_ckpt", default=None, help="预训练 DeepKAN 的 .pth（state_dict）路径")

    parser.add_argument("--threshold_non_toxic", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="基因数大时 attention 显存约 O(num_gene^2)，易 OOM，默认 1；显存够可试 2/4/8")
    parser.add_argument("--max_epoch", type=int, default=30)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--kan_knots", type=int, default=5)
    parser.add_argument("--pool", choices=["mean", "meanmax"], default="meanmax")
    parser.add_argument("--freeze_deepce", action="store_true", help="冻结 DeepKAN（推荐：有预训练权重时）")

    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=343)
    parser.add_argument("--out_dir", default="tox_out")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_spec = ToxLabelSpec(threshold_non_toxic=float(args.threshold_non_toxic))
    full_ds = ToxicityCSVDataset(args.tox_csv, label_spec=label_spec)
    train_idx, dev_idx, test_idx = split_indices_stratified(
        [full_ds[i]["label"] for i in range(len(full_ds))],
        dev_ratio=float(args.dev_ratio),
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
    )

    train_ds = torch.utils.data.Subset(full_ds, train_idx.tolist())
    dev_ds = torch.utils.data.Subset(full_ds, dev_idx.tolist())
    test_ds = torch.utils.data.Subset(full_ds, test_idx.tolist())

    # gene/cell
    gene = read_gene(args.gene_file, device=device).float()  # [G, D]
    cell_features = None
    include_cell_id = False
    default_cell_idx = 0

    # 若提供了 DeepCE checkpoint，先检查其是否带 cell_id（与训练时一致才能正确加载）
    deepce_state_for_infer = None
    if args.deepce_ckpt is not None:
        ckpt_raw = torch.load(args.deepce_ckpt, map_location=device)
        deepce_state_for_infer = ckpt_raw.get("model_state_dict") or ckpt_raw.get("state_dict") or ckpt_raw
        if not isinstance(deepce_state_for_infer, dict):
            deepce_state_for_infer = None

    # 检测 checkpoint 是否含 cell：用 key 名或 linear_1 的 input 维数推断（260 = 256+4 表示带 cell）
    ckpt_has_cell = False
    if deepce_state_for_infer is not None:
        if "cell_id_embed.weight" in deepce_state_for_infer:
            ckpt_has_cell = True
        else:
            # 无 cell_id 键时用 linear_1.coef 第一维判断：260 表示带 cell
            coef = deepce_state_for_infer.get("linear_1.coef")
            if coef is not None and coef.dim() >= 1 and int(coef.shape[0]) == 260:
                ckpt_has_cell = True

    if args.cell_file is not None:
        cell_features, cell_id_to_idx = read_cell(args.cell_file, device=device)
        include_cell_id = True
        if args.default_cell_id is not None:
            if args.default_cell_id not in cell_id_to_idx:
                raise ValueError(
                    f"default_cell_id={args.default_cell_id} 不在 cell_file 中。可用示例: {list(cell_id_to_idx.keys())[:10]}"
                )
            default_cell_idx = int(cell_id_to_idx[args.default_cell_id])
    elif ckpt_has_cell:
        # checkpoint 带 cell 头，但未传 cell_file：用虚拟 cell 使结构一致以便加载
        if "cell_id_embed.weight" in (deepce_state_for_infer or {}):
            w = deepce_state_for_infer["cell_id_embed.weight"]
            cell_id_input_dim_ckpt = int(w.shape[1])
        else:
            cell_id_input_dim_ckpt = 735  # 与 predict_deepce 中 cell 特征维常见值
        cell_features = torch.zeros(1, cell_id_input_dim_ckpt, device=device, dtype=torch.float32)
        include_cell_id = True
        default_cell_idx = 0
        print(f"[DeepCE] checkpoint 含 cell_id，已用虚拟 cell (1, {cell_id_input_dim_ckpt}) 以匹配结构")

    train_loader = create_dataloader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        device=device,
        include_cell_id=include_cell_id,
        default_cell_idx=default_cell_idx,
    )
    dev_loader = create_dataloader(
        dev_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        device=device,
        include_cell_id=include_cell_id,
        default_cell_idx=default_cell_idx,
    )
    test_loader = create_dataloader(
        test_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        device=device,
        include_cell_id=include_cell_id,
        default_cell_idx=default_cell_idx,
    )

    # DeepCE base（cell 维度须与 checkpoint 一致）
    cell_id_emb_dim = 4
    if cell_features is not None and deepce_state_for_infer is not None and "cell_id_embed.weight" in deepce_state_for_infer:
        cell_id_emb_dim = int(deepce_state_for_infer["cell_id_embed.weight"].shape[0])
    deepce = DeepCE(
        drug_input_dim={"atom": 62, "bond": 6},
        drug_emb_dim=128,
        conv_size=[16, 16],
        degree=[0, 1, 2, 3, 4, 5],
        gene_input_dim=int(gene.shape[1]),
        gene_emb_dim=128,
        num_gene=int(gene.shape[0]),
        hid_dim=128,
        dropout=float(args.dropout),
        loss_type="cross_entropy",
        device=device,
        initializer=torch.nn.init.xavier_uniform_,
        cell_id_input_dim=int(cell_features.shape[1]) if cell_features is not None else 0,
        cell_id_emb_dim=cell_id_emb_dim,
        use_cell_id=include_cell_id,
    ).float().to(device)

    if args.deepce_ckpt is not None:
        state = deepce_state_for_infer if deepce_state_for_infer is not None else torch.load(args.deepce_ckpt, map_location=device)
        if isinstance(state, dict):
            state = state.get("model_state_dict") or state.get("state_dict") or state
        deepce.load_state_dict(state, strict=True)

    head_cfg = ToxicityHeadConfig(
        num_classes=5,
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
        kan_knots=int(args.kan_knots),
        pool=str(args.pool),
        use_probs=True,
    )
    model = DeepCETwitteroxicityKAN(deepce=deepce, head_cfg=head_cfg, freeze_deepce=bool(args.freeze_deepce)).to(device)

    # optimizer: 只优化 requires_grad=True 的参数
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay))
    criterion = nn.CrossEntropyLoss()

    best_dev_f1 = -1.0
    best_path = out_dir / "best_toxicity_kan.pth"
    start = datetime.now()

    def run_eval(loader, name: str):
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
        rep = classification_report(y_true, y_pred, target_names=list(label_spec.label_names), digits=4)
        print(f"\n===== {name} =====")
        print(f"ACC={acc:.4f} | Macro-F1={f1:.4f}")
        print(rep)
        return acc, f1

    for epoch in range(int(args.max_epoch)):
        model.train()
        losses = []
        for ft, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                drug=ft["drug"],
                gene=gene,
                mask=ft["mask"],
                cell_id=ft.get("cell_id"),
                cell_features=cell_features,
            )
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

        print(f"Epoch {epoch+1:03d} | Train Loss={np.mean(losses):.4f}")
        _, dev_f1 = run_eval(dev_loader, "DEV")
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            payload = {
                "model_state_dict": model.state_dict(),
                "head_cfg": asdict(head_cfg),
                "label_spec": asdict(label_spec),
                "gene_file": args.gene_file,
                "cell_file": args.cell_file,
                "default_cell_id": args.default_cell_id,
                "deepce_ckpt": args.deepce_ckpt,
                "created_at": start.isoformat(),
            }
            torch.save(payload, best_path)
            meta = {k: v for k, v in payload.items() if k != "model_state_dict"}
            (out_dir / "best_meta.json").write_text(
                json.dumps(meta, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
            print(f"--- 保存最佳模型: {best_path} (Macro-F1={best_dev_f1:.4f}) ---")

    print(f"\n训练完成，用时: {datetime.now() - start}")
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    run_eval(test_loader, "TEST")


if __name__ == "__main__":
    main()
    # 不带 cell 和 deepce：
#   python main_toxicity.py --tox_csv data/log_normalized_data.csv --gene_file data/DeepCE_gene_vectors.csv
# 使用 main_deepce.py 保存的 DeepCE 权重（默认保存到 deepce_out/deepce_best.pth）：
#   python main_toxicity.py --tox_csv data/log_normalized_data.csv --gene_file data/DeepCE_gene_vectors.csv --deepce_ckpt deepce_out/deepce_best.pth --freeze_deepce
