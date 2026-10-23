import os
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
import argparse
from datetime import datetime
from models import DeepCE
from utils import DataReader, classification_metrics
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt # 如果需要画图
start_time = datetime.now()
parser = argparse.ArgumentParser(description='DeepCE 三分类训练')
parser.add_argument('--drug_file')
parser.add_argument('--gene_file')
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--train_file')
parser.add_argument('--dev_file')
parser.add_argument('--test_file')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--max_epoch', type=int, default=50)
parser.add_argument('--cell_file')
parser.add_argument('--cell_ids', type=str, default=None,
                    help='可选，逗号分隔的 cell_id 列表，例如 \"A375,MCF7,HT29\"，只使用这些细胞的数据')
parser.add_argument('--save_dir', default='deepce_out', help='保存 DeepCE checkpoint 的目录')
parser.add_argument('--save_best', action='store_true', help='在 dev 指标最优时保存 best checkpoint')
parser.add_argument('--save_last', action='store_true', help='训练结束保存 last checkpoint')
parser.add_argument('--best_metric', choices=['auc', 'f1', 'acc'], default='auc', help='best 的选择指标（基于 dev）')
parser.add_argument('--grad_accum_steps', type=int, default=1, help='梯度累积步数')
parser.add_argument('--grad_clip', type=float, default=0.0, help='梯度裁剪阈值，<=0 表示关闭')
parser.add_argument('--lr', type=float, default=2e-4, help='学习率')
parser.add_argument('--weight_decay', type=float, default=1e-2, help='AdamW 权重衰减')
parser.add_argument('--loss_mode', type=str, default='focal+penalty',
                    choices=['ce', 'weighted_ce', 'focal', 'focal+penalty'],
                    help='损失模式')
parser.add_argument('--loss_down_weight', type=float, default=1.8, help='类别0(Down)损失权重')
parser.add_argument('--loss_none_weight', type=float, default=0.75, help='类别1(None)损失权重')
parser.add_argument('--loss_up_weight', type=float, default=1.8, help='类别2(Up)损失权重')
parser.add_argument('--loss_focal_gamma', type=float, default=2.0, help='Focal Loss gamma')
parser.add_argument('--loss_down_up_miss_scale', type=float, default=1.5,
                    help='focal+penalty下，Down/Up被预测为None的额外惩罚系数')
parser.add_argument('--train_sampler', type=str, default='weighted', choices=['random', 'weighted'],
                    help='训练集采样策略')
parser.add_argument('--sampler_power', type=float, default=1.0,
                    help='weighted采样强度，越大越偏向少数类')
parser.add_argument('--class_weight_beta', type=float, default=0.9999,
                    help='基于有效样本数的类别权重 beta，越接近1越强调稀有类')
parser.add_argument('--class_weight_scale', type=float, default=1.0,
                    help='自动类别权重整体倍率系数，>1 放大少数类权重，<1 缩小权重')
parser.add_argument('--manual_class_weights', type=str, default=None,
                    help='手动指定类别权重，格式如 2.0,1.0,2.0；指定后会覆盖自动权重')
parser.add_argument('--seed', type=int, default=343, help='随机种子，控制可复现实验')
parser.add_argument('--plot_tsne', action='store_true', help='是否绘制细胞特征t-SNE（默认关闭）')
parser.add_argument('--deg_topk', type=str, default='50,100,200',
                    help='DEGs recovery的top-k列表，逗号分隔，如 50,100,200')
parser.add_argument('--metric_chunk_size', type=int, default=2048,
                    help='评估时按基因维度分块计算指标的大小，越小越省内存')
parser.add_argument('--attn_use_global_context', action='store_true',
                    help='启用全局环境感知（GAP+广播拼接）')
parser.add_argument('--attn_disable_global_context', action='store_true',
                    help='关闭全局环境感知，退化为局部独立预测（baseline）')
parser.add_argument('--attn_use_masked_gap', action='store_true',
                    help='在全局池化时使用 mask（忽略padding基因）')
parser.add_argument('--attn_use_fusion_projection', action='store_true',
                    help='拼接后增加投影层(2H->H)+GELU 再进入输出头')
parser.add_argument('--attn_global_context_alpha', type=float, default=1.0,
                    help='全局上下文缩放系数 alpha')
parser.set_defaults(attn_use_global_context=True)
args = parser.parse_args()

# 固定随机种子，保证多 seed 对比可复现
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(args.seed)
print(f"Random Seed | seed={args.seed}")

# 初始化参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = args.batch_size
max_epoch = args.max_epoch
accumulation_steps = max(1, args.grad_accum_steps)
num_gene = 11300  # 请根据 gene_file 实际行数调整

# 根据命令行参数构造 cell 过滤条件
filter_cfg = {}
if args.cell_ids:
    cell_list = [c.strip() for c in args.cell_ids.split(',') if c.strip()]
    if len(cell_list) > 0:
        filter_cfg['cell_id'] = cell_list

if args.cell_file is None:
    raise ValueError("This pipeline now requires --cell_file because cell features are mandatory.")

data = DataReader(args.drug_file, args.gene_file, args.train_file, args.dev_file,
                  args.test_file, filter=filter_cfg, device=device, cell_file=args.cell_file,
                  train_sampler=args.train_sampler, sampler_power=args.sampler_power,
                  class_weight_beta=args.class_weight_beta)
print(f"Cell File | {args.cell_file}")

model = DeepCE(drug_input_dim={'atom': 62, 'bond': 6}, drug_emb_dim=128,
               conv_size=[16, 16], degree=[0, 1, 2, 3, 4, 5],
               gene_input_dim=np.shape(data.gene)[1], gene_emb_dim=128,
               num_gene=np.shape(data.gene)[0], hid_dim=128, dropout=float(args.dropout),
               loss_type=args.loss_mode, device=device, initializer=torch.nn.init.xavier_uniform_,
               cell_id_input_dim=np.shape(data.cell)[1] if data.cell is not None else 0,
               cell_id_emb_dim=4, use_cell_id=True,
               attn_use_global_context=bool(args.attn_use_global_context and not args.attn_disable_global_context),
               attn_use_masked_gap=bool(args.attn_use_masked_gap),
               attn_use_fusion_projection=bool(args.attn_use_fusion_projection),
               attn_global_context_alpha=float(args.attn_global_context_alpha))
auto_train_class_weights = data.train_class_weights.detach().cpu()
scaled_train_class_weights = auto_train_class_weights * float(args.class_weight_scale)

# 权重优先级：
# 1) 如果显式传入了 loss_down/none/up_weight，则优先使用
# 2) 否则如果传入了 manual_class_weights，则使用手动权重
# 3) 否则使用 DataReader 计算得到的自动权重（及其 scale）
explicit_loss_weights = [args.loss_down_weight, args.loss_none_weight, args.loss_up_weight]
has_explicit_loss_weights = any(
    w is not None for w in explicit_loss_weights
)

if has_explicit_loss_weights:
    if any(w is None for w in explicit_loss_weights):
        raise ValueError(
            "--loss_down_weight, --loss_none_weight, and --loss_up_weight must be provided together"
        )
    train_class_weights = torch.tensor(explicit_loss_weights, dtype=torch.float32)
    weight_source = 'explicit_loss_args'
elif args.manual_class_weights is not None:
    manual_weights = [float(x.strip()) for x in args.manual_class_weights.split(',') if x.strip()]
    if len(manual_weights) != 3:
        raise ValueError("--manual_class_weights must contain exactly 3 values: down, none, up")
    train_class_weights = torch.tensor(manual_weights, dtype=torch.float32)
    weight_source = 'manual'
else:
    train_class_weights = scaled_train_class_weights
    weight_source = 'auto'

model.loss_down_weight = float(train_class_weights[0].item())
model.loss_none_weight = float(train_class_weights[1].item())
model.loss_up_weight = float(train_class_weights[2].item())
model.loss_focal_gamma = args.loss_focal_gamma
model.loss_down_up_miss_scale = args.loss_down_up_miss_scale

print(f"Auto Train Class Weights | {auto_train_class_weights.tolist()}")
print(f"Scaled Train Class Weights | scale={args.class_weight_scale}, weights={scaled_train_class_weights.tolist()}")
print(f"Final Train Class Weights | source={weight_source}, weights={train_class_weights.tolist()}")

model = model.float().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

best_dev_acc = 0.0
best_dev_auc = -1.0
best_dev_f1 = -1.0

os.makedirs(args.save_dir, exist_ok=True)
best_path = os.path.join(args.save_dir, 'deepce_best.pth')
last_path = os.path.join(args.save_dir, 'deepce_last.pth')


# --- 新增：细胞系特征可视化部分 ---
def plot_cell_tsne(cell_features, cell_names):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    print("\n正在生成细胞系 t-SNE 可视化...")

    # 转换为 CPU numpy 数组
    if torch.is_tensor(cell_features):
        X = cell_features.detach().cpu().numpy()
    else:
        X = cell_features

    # 11个样本，perplexity 必须很小
    tsne = TSNE(n_components=2, perplexity=3, init='pca', random_state=42, learning_rate='auto')
    X_embedded = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))
    for i, name in enumerate(cell_names):
        plt.scatter(X_embedded[i, 0], X_embedded[i, 1], s=100)
        plt.annotate(name, (X_embedded[i, 0], X_embedded[i, 1]), xytext=(5, 2),
                     textcoords='offset points', fontsize=9)

    plt.title(f't-SNE visualization of {len(cell_names)} Cell Lines')
    plt.grid(True, linestyle='--', alpha=0.6)

    # 保存图片到本地，防止服务器无法直接弹窗
    plt.savefig('cell_tsne.png')
    print("t-SNE 图片已保存为 'cell_tsne.png'")
    # plt.show() # 如果在有 GUI 的环境下可以开启


# 提取细胞系 ID 列表（假设 data 里面存储了 cell 列表）
# 这里需要根据你 DataReader 里的实现获取 cell_names
# 如果没有直接的列表，可能需要从 args.cell_file 读取
cell_names = ['BT20', 'MCF7', 'MDAMB231', 'SKBR3', 'A375','A549','HCC515','HT29','JURKAT','PC3','U2OS']

if args.plot_tsne and data.cell is not None:
    # 注意：data.cell 形状应该是 [11, 735]
    plot_cell_tsne(data.cell, cell_names)
# --- 可视化部分结束 ---

effective_batch_size = batch_size * accumulation_steps
print(f"Train Config | batch_size={batch_size}, grad_accum_steps={accumulation_steps}, effective_batch_size={effective_batch_size}")
print(f"Optimizer | AdamW(lr={args.lr}, weight_decay={args.weight_decay}), grad_clip={args.grad_clip}")
print(
    "Imbalance Strategy | "
    f"loss_mode={args.loss_mode}, auto_class_weights={train_class_weights.tolist()}, "
    f"focal_gamma={args.loss_focal_gamma}, down_up_miss_scale={args.loss_down_up_miss_scale}, "
    f"train_sampler={args.train_sampler}, sampler_power={args.sampler_power}, "
    f"class_weight_beta={args.class_weight_beta}"
)
print(
    "Global Context Ablation | "
    f"use_global_context={bool(args.attn_use_global_context and not args.attn_disable_global_context)}, "
    f"use_masked_gap={bool(args.attn_use_masked_gap)}, "
    f"use_fusion_projection={bool(args.attn_use_fusion_projection)}, "
    f"global_context_alpha={float(args.attn_global_context_alpha)}"
)


def parse_topk_list(topk_text):
    values = []
    for token in str(topk_text).split(','):
        token = token.strip()
        if not token:
            continue
        k = int(token)
        if k > 0:
            values.append(k)
    if not values:
        raise ValueError(f"Invalid --deg_topk: {topk_text}")
    return values


deg_topk_list = parse_topk_list(args.deg_topk)
print(f"Evaluation Extra | deg_topk={deg_topk_list}")
train_label_flat = data.train_label.detach().long().view(-1).cpu()
class_count = torch.bincount(train_label_flat, minlength=3).tolist()
print(f"Train Label Distribution | Down={class_count[0]}, None={class_count[1]}, Up={class_count[2]}")
data.gene = data.gene.float()
data.cell = data.cell.float() if data.cell is not None else None


def safe_macro_auc(y_true_flat, y_prob_flat):
    unique_classes = np.unique(y_true_flat)
    if unique_classes.shape[0] < 2:
        return None, f"only {unique_classes.shape[0]} class present"
    try:
        auc = roc_auc_score(y_true_flat, y_prob_flat, multi_class='ovr', average='macro')
        return float(auc), None
    except ValueError as err:
        return None, str(err)


def safe_pearson_global(y_true_flat, y_pred_flat):
    y_true_flat = np.asarray(y_true_flat, dtype=np.float64)
    y_pred_flat = np.asarray(y_pred_flat, dtype=np.float64)
    if y_true_flat.size == 0 or y_pred_flat.size == 0:
        return None, "empty input"
    if np.std(y_true_flat) < 1e-12 or np.std(y_pred_flat) < 1e-12:
        return None, "constant vector"
    value = np.corrcoef(y_true_flat, y_pred_flat)[0, 1]
    if np.isnan(value):
        return None, "nan result"
    return float(value), None


def safe_pearson_per_sample(y_true_matrix, y_pred_matrix):
    y_true_matrix = np.asarray(y_true_matrix, dtype=np.float64)
    y_pred_matrix = np.asarray(y_pred_matrix, dtype=np.float64)
    if y_true_matrix.shape != y_pred_matrix.shape:
        return None, "shape mismatch"
    if y_true_matrix.ndim != 2:
        return None, "expected 2D matrix"
    values = []
    for idx in range(y_true_matrix.shape[0]):
        yt = y_true_matrix[idx]
        yp = y_pred_matrix[idx]
        if np.std(yt) < 1e-12 or np.std(yp) < 1e-12:
            continue
        corr = np.corrcoef(yt, yp)[0, 1]
        if not np.isnan(corr):
            values.append(float(corr))
    if not values:
        return None, "no valid sample (constant vectors)"
    return float(np.mean(values)), None


def safe_mse(y_true_flat, y_pred_flat):
    y_true_flat = np.asarray(y_true_flat, dtype=np.float64)
    y_pred_flat = np.asarray(y_pred_flat, dtype=np.float64)
    if y_true_flat.size == 0 or y_pred_flat.size == 0:
        return None, "empty input"
    if y_true_flat.shape != y_pred_flat.shape:
        return None, "shape mismatch"
    return float(np.mean((y_true_flat - y_pred_flat) ** 2)), None


def _safe_nanmean(values):
    values = [v for v in values if np.isfinite(v)]
    return float(np.mean(values)) if values else float('nan')


def _rankdata_1d(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    return ranks


def safe_spearman_global(y_true_flat, y_pred_flat):
    y_true_flat = np.asarray(y_true_flat, dtype=np.float64)
    y_pred_flat = np.asarray(y_pred_flat, dtype=np.float64)
    if y_true_flat.size == 0 or y_pred_flat.size == 0:
        return None, "empty input"
    if y_true_flat.shape != y_pred_flat.shape:
        return None, "shape mismatch"
    r_true = _rankdata_1d(y_true_flat)
    r_pred = _rankdata_1d(y_pred_flat)
    if np.std(r_true) < 1e-12 or np.std(r_pred) < 1e-12:
        return None, "constant rank vector"
    value = np.corrcoef(r_true, r_pred)[0, 1]
    if np.isnan(value):
        return None, "nan result"
    return float(value), None


def safe_spearman_per_sample(y_true_matrix, y_pred_matrix):
    y_true_matrix = np.asarray(y_true_matrix, dtype=np.float64)
    y_pred_matrix = np.asarray(y_pred_matrix, dtype=np.float64)
    if y_true_matrix.shape != y_pred_matrix.shape:
        return None, "shape mismatch"
    if y_true_matrix.ndim != 2:
        return None, "expected 2D matrix"
    values = []
    for idx in range(y_true_matrix.shape[0]):
        yt = y_true_matrix[idx]
        yp = y_pred_matrix[idx]
        r_true = _rankdata_1d(yt)
        r_pred = _rankdata_1d(yp)
        if np.std(r_true) < 1e-12 or np.std(r_pred) < 1e-12:
            continue
        corr = np.corrcoef(r_true, r_pred)[0, 1]
        if not np.isnan(corr):
            values.append(float(corr))
    if not values:
        return None, "no valid sample"
    return float(np.mean(values)), None


def degs_recovery_metrics(y_true_matrix, y_prob_matrix, topk_list):
    # y_true_matrix: [B, G], labels 0/1/2
    # y_prob_matrix: [B, G, 3], class probs for Down/None/Up
    y_true_matrix = np.asarray(y_true_matrix)
    y_prob_matrix = np.asarray(y_prob_matrix)
    if y_true_matrix.ndim != 2 or y_prob_matrix.ndim != 3:
        return {}

    num_sample, num_gene = y_true_matrix.shape
    out = {}
    for k in topk_list:
        k_eff = min(int(k), int(num_gene))
        if k_eff <= 0:
            continue
        up_precisions, up_recalls = [], []
        down_precisions, down_recalls = [], []

        for i in range(num_sample):
            yt = y_true_matrix[i]
            prob_down = y_prob_matrix[i, :, 0]
            prob_up = y_prob_matrix[i, :, 2]

            true_up = set(np.where(yt == 2)[0].tolist())
            true_down = set(np.where(yt == 0)[0].tolist())

            pred_up = set(np.argsort(prob_up)[-k_eff:].tolist())
            pred_down = set(np.argsort(prob_down)[-k_eff:].tolist())

            if len(true_up) > 0:
                hit_up = len(pred_up & true_up)
                up_precisions.append(hit_up / k_eff)
                up_recalls.append(hit_up / len(true_up))
            if len(true_down) > 0:
                hit_down = len(pred_down & true_down)
                down_precisions.append(hit_down / k_eff)
                down_recalls.append(hit_down / len(true_down))

        def _mean_or_nan(vals):
            return float(np.mean(vals)) if vals else float('nan')

        out[k_eff] = {
            'up_precision': _mean_or_nan(up_precisions),
            'up_recall': _mean_or_nan(up_recalls),
            'down_precision': _mean_or_nan(down_precisions),
            'down_recall': _mean_or_nan(down_recalls),
        }
    return out

def safe_roc_auc_binary(y_true_binary, y_score):
    y_true_binary = np.asarray(y_true_binary)
    y_score = np.asarray(y_score)
    if y_true_binary.size == 0 or y_score.size == 0:
        return None, 'empty input'
    if np.unique(y_true_binary).shape[0] < 2:
        return None, 'only one class present'
    try:
        return float(roc_auc_score(y_true_binary, y_score)), None
    except ValueError as err:
        return None, str(err)


def collect_eval_metrics_in_chunks(model, phase, batch_size, data, deg_topk_list):
    total_samples = 0
    correct = 0
    y_true_chunks = []
    y_pred_chunks = []
    prob_chunks = []
    sample_true_rows = []
    sample_pred_rows = []
    sample_prob_rows = []
    true_dist = np.zeros(3, dtype=np.int64)
    pred_dist = np.zeros(3, dtype=np.int64)

    with torch.no_grad():
        for _, b in enumerate(data.get_batch_data(dataset=phase, batch_size=batch_size, shuffle=False)):
            f, l = b
            pred_gene, _ = model(f['drug'], data.gene, f['mask'], f.get('cell_id'), data.cell, return_overall=True)
            pred_cls = torch.argmax(pred_gene, dim=-1)
            pred_prob = torch.softmax(pred_gene, dim=-1)

            y_true = l.detach().cpu().numpy()
            y_pred = pred_cls.detach().cpu().numpy()
            y_prob = pred_prob.detach().cpu().numpy()

            y_true_flat = y_true.ravel()
            y_pred_flat = y_pred.ravel()
            y_prob_flat = y_prob.reshape(-1, 3)

            total_samples += y_true_flat.size
            correct += int((y_true_flat == y_pred_flat).sum())
            true_dist += np.bincount(y_true_flat.astype(np.int64), minlength=3)
            pred_dist += np.bincount(y_pred_flat.astype(np.int64), minlength=3)

            y_true_chunks.append(y_true_flat)
            y_pred_chunks.append(y_pred_flat)
            prob_chunks.append(y_prob_flat)
            sample_true_rows.extend(list(y_true))
            sample_pred_rows.extend(list(y_pred))
            sample_prob_rows.extend(list(y_prob))

    if total_samples == 0:
        return None

    y_t_flat = np.concatenate(y_true_chunks, axis=0)
    y_p_flat = np.concatenate(y_pred_chunks, axis=0)
    y_prob_flat = np.concatenate(prob_chunks, axis=0)
    Y_t_matrix = np.asarray(sample_true_rows)
    Y_p_matrix = np.asarray(sample_pred_rows)
    Y_prob_matrix = np.asarray(sample_prob_rows)

    true_ratio = true_dist / max(true_dist.sum(), 1)
    pred_ratio = pred_dist / max(pred_dist.sum(), 1)

    acc, f1, report = classification_metrics(y_t_flat, y_p_flat)
    macro_auc, macro_auc_err = safe_macro_auc(y_t_flat, y_prob_flat)
    pearson_global, pearson_global_err = safe_pearson_global(y_t_flat, y_p_flat)
    pearson_sample, pearson_sample_err = safe_pearson_per_sample(Y_t_matrix, Y_p_matrix)
    spearman_global, spearman_global_err = safe_spearman_global(y_t_flat, y_p_flat)
    spearman_sample, spearman_sample_err = safe_spearman_per_sample(Y_t_matrix, Y_p_matrix)
    mse_value, mse_err = safe_mse(y_t_flat, y_p_flat)
    deg_metrics = degs_recovery_metrics(Y_t_matrix, Y_prob_matrix, deg_topk_list)

    return {
        'acc': acc,
        'f1': f1,
        'report': report,
        'macro_auc': macro_auc,
        'macro_auc_err': macro_auc_err,
        'pearson_global': pearson_global,
        'pearson_global_err': pearson_global_err,
        'pearson_sample': pearson_sample,
        'pearson_sample_err': pearson_sample_err,
        'spearman_global': spearman_global,
        'spearman_global_err': spearman_global_err,
        'spearman_sample': spearman_sample,
        'spearman_sample_err': spearman_sample_err,
        'mse_value': mse_value,
        'mse_err': mse_err,
        'deg_metrics': deg_metrics,
        'true_dist': true_dist,
        'pred_dist': pred_dist,
        'true_ratio': true_ratio,
        'pred_ratio': pred_ratio,
        'y_t_flat': y_t_flat,
        'y_p_flat': y_p_flat,
        'y_prob_flat': y_prob_flat,
        'Y_t_matrix': Y_t_matrix,
        'Y_p_matrix': Y_p_matrix,
        'Y_prob_matrix': Y_prob_matrix,
    }


for epoch in range(max_epoch):
    model.train()
    epoch_loss = 0
    optimizer.zero_grad()
    num_train_steps = 0
    for i, batch in enumerate(data.get_batch_data(dataset='train', batch_size=batch_size, shuffle=True)):
        ft, lb = batch
        num_train_steps += 1

        # 强制转换类型
        if isinstance(ft['drug'], dict):
            ft['drug']['atom'] = ft['drug']['atom'].float()
            ft['drug']['bond'] = ft['drug']['bond'].float()

        predict_gene, predict_overall = model(ft['drug'], data.gene, ft['mask'], ft.get('cell_id'), data.cell, return_overall=True)

        # 第一阶段改进：用更稳健的 Focal Loss / Weighted CE 抑制预测塌缩
        loss = model.loss(lb.long(), predict_gene)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Epoch {epoch + 1} Step {i + 1} | Skip invalid loss: {loss.item()}")
            optimizer.zero_grad()
            continue
        loss = loss / accumulation_steps
        loss.backward()
        if (i + 1) % accumulation_steps == 0:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
        epoch_loss += loss.item() * accumulation_steps

    # 处理梯度累积的尾批次，防止剩余梯度丢失
    if num_train_steps > 0 and (num_train_steps % accumulation_steps != 0):
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
        optimizer.step()
        optimizer.zero_grad()

    if num_train_steps == 0:
        print(f"Epoch {epoch + 1} | Train Loss: N/A (empty train loader)")
        continue
    if num_train_steps > 0 and np.isfinite(epoch_loss):
        avg_train_loss = epoch_loss / num_train_steps
    else:
        avg_train_loss = float('nan')
    print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f}")

    # ... 前面代码保持不变 ...
    # 评估逻辑
    model.eval()
    for phase in ['dev', 'test']:
        metrics = collect_eval_metrics_in_chunks(model, phase, batch_size, data, deg_topk_list)
        if metrics is None:
            print(f"\n===== {phase.upper()} RESULT =====")
            print("No evaluation samples found.")
            continue

        y_t_flat = metrics['y_t_flat']
        y_p_flat = metrics['y_p_flat']
        y_prob_flat = metrics['y_prob_flat']
        Y_t_matrix = metrics['Y_t_matrix']
        Y_p_matrix = metrics['Y_p_matrix']

        print(f"\n===== {phase.upper()} RESULT =====")
        macro_auc_str = f"{metrics['macro_auc']:.4f}" if metrics['macro_auc'] is not None else "N/A"
        pearson_global_str = f"{metrics['pearson_global']:.4f}" if metrics['pearson_global'] is not None else "N/A"
        pearson_sample_str = f"{metrics['pearson_sample']:.4f}" if metrics['pearson_sample'] is not None else "N/A"
        spearman_global_str = f"{metrics['spearman_global']:.4f}" if metrics['spearman_global'] is not None else "N/A"
        spearman_sample_str = f"{metrics['spearman_sample']:.4f}" if metrics['spearman_sample'] is not None else "N/A"
        mse_str = f"{metrics['mse_value']:.4f}" if metrics['mse_value'] is not None else "N/A"
        print(
            f"Accuracy: {metrics['acc']:.4f} | Macro-F1: {metrics['f1']:.4f} | Macro-AUC: {macro_auc_str} | "
            f"PCC-Global: {pearson_global_str} | PCC-SampleMean: {pearson_sample_str} | "
            f"Spearman-Global: {spearman_global_str} | Spearman-SampleMean: {spearman_sample_str} | "
            f"MSE: {mse_str}"
        )
        print(
            "Label/Pred Dist | "
            f"True={metrics['true_dist'].tolist()} ({[round(x, 4) for x in metrics['true_ratio'].tolist()]}), "
            f"Pred={metrics['pred_dist'].tolist()} ({[round(x, 4) for x in metrics['pred_ratio'].tolist()]})"
        )
        if metrics['macro_auc_err']:
            print(f"Macro-AUC skipped: {metrics['macro_auc_err']}")
        if metrics['pearson_global_err']:
            print(f"Pearson-Global skipped: {metrics['pearson_global_err']}")
        if metrics['pearson_sample_err']:
            print(f"Pearson-SampleMean skipped: {metrics['pearson_sample_err']}")
        if metrics['spearman_global_err']:
            print(f"Spearman-Global skipped: {metrics['spearman_global_err']}")
        if metrics['spearman_sample_err']:
            print(f"Spearman-SampleMean skipped: {metrics['spearman_sample_err']}")
        if metrics['mse_err']:
            print(f"MSE skipped: {metrics['mse_err']}")
        if metrics['deg_metrics']:
            print("DEGs Recovery:")
            for k in sorted(metrics['deg_metrics'].keys()):
                m = metrics['deg_metrics'][k]
                print(
                    f"  Top-{k} | "
                    f"Up(P={m['up_precision']:.4f}, R={m['up_recall']:.4f}) | "
                    f"Down(P={m['down_precision']:.4f}, R={m['down_recall']:.4f})"
                )
        print("Detailed Report:")
        print(metrics['report'])

        # 修改点 2：如果是测试集，显示前 5 个样本的前 20 个基因的预测 vs 真实
        if phase == 'test':
            # 打印各类的单独 AUC
            for i, label_name in enumerate(['Down', 'None', 'Up']):
                class_binary = (y_t_flat == i).astype(int)
                if np.unique(class_binary).shape[0] < 2:
                    print(f"Class {label_name} AUC: N/A (class missing in ground truth)")
                    continue
                try:
                    class_auc = roc_auc_score(class_binary, y_prob_flat[:, i])
                    print(f"Class {label_name} AUC: {class_auc:.4f}")
                except ValueError as err:
                    print(f"Class {label_name} AUC: N/A ({err})")
            print("\nSample Predictions (First 5 samples, first 20 genes):")
            max_samples_to_show = min(5, Y_t_matrix.shape[0])
            for sample_idx in range(max_samples_to_show):
                print(f"Sample {sample_idx} | True: {Y_t_matrix[sample_idx, :20].astype(int)}")
                print(f"Sample {sample_idx} | Pred: {Y_p_matrix[sample_idx, :20].astype(int)}")
            unique, counts = np.unique(y_p_flat, return_counts=True)
            print("Prediction Distribution (Class: Count):", dict(zip(unique, counts)))

        if phase == 'dev':
            improved = False
            if args.best_metric == 'auc' and metrics['macro_auc'] is not None and metrics['macro_auc'] > best_dev_auc:
                best_dev_auc = metrics['macro_auc']
                improved = True
            elif args.best_metric == 'f1' and metrics['f1'] > best_dev_f1:
                best_dev_f1 = metrics['f1']
                improved = True
            elif args.best_metric == 'acc' and metrics['acc'] > best_dev_acc:
                best_dev_acc = metrics['acc']
                improved = True

            pred_majority_ratio = float(metrics['pred_ratio'].max())
            if pred_majority_ratio > 0.85:
                majority_class = int(metrics['pred_ratio'].argmax())
                print(
                    f"[Warn] Potential collapse on dev: majority_pred_class={majority_class}, "
                    f"ratio={pred_majority_ratio:.4f}, pred_dist={metrics['pred_dist'].tolist()}"
                )

            if improved and args.save_best:
                ckpt = {
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'dev_acc': float(metrics['acc']),
                    'dev_f1': float(metrics['f1']),
                    'dev_macro_auc': float(metrics['macro_auc']) if metrics['macro_auc'] is not None else float('nan'),
                    'dev_pearson_global': float(metrics['pearson_global']) if metrics['pearson_global'] is not None else float('nan'),
                    'dev_pearson_sample_mean': float(metrics['pearson_sample']) if metrics['pearson_sample'] is not None else float('nan'),
                    'dev_spearman_global': float(metrics['spearman_global']) if metrics['spearman_global'] is not None else float('nan'),
                    'dev_spearman_sample_mean': float(metrics['spearman_sample']) if metrics['spearman_sample'] is not None else float('nan'),
                    'dev_mse': float(metrics['mse_value']) if metrics['mse_value'] is not None else float('nan'),
                    'dev_deg_recovery': metrics['deg_metrics'],
                    'args': vars(args),
                }
                torch.save(ckpt, best_path)
                print(f"--- Found Best Model (dev {args.best_metric}), saved to {best_path} ---")

print(f"Total Time: {datetime.now() - start_time}")

if args.save_last:
    ckpt = {
        'model_state_dict': model.state_dict(),
        'epoch': max_epoch,
        'best_dev_acc': float(best_dev_acc),
        'best_dev_f1': float(best_dev_f1),
        'best_dev_macro_auc': float(best_dev_auc),
        'args': vars(args),
    }
    torch.save(ckpt, last_path)
    print(f"Saved last checkpoint to {last_path}")