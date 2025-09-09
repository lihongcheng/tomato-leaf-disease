import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from data_process import DataProcessor, PlantDataset
from model_def import MobileNetV3_CHAM_EarlyExit

# 路径配置（与训练脚本对齐）
RESULTS_DIR = "./results/MobileNetV3_CHAM_EarlyExit"
CHECKPOINT_DIR = "./checkpoints/MobileNetV3_CHAM_EarlyExit"
BEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_checkpoint.pth")
OUT_CSV = os.path.join(RESULTS_DIR, "threshold_search_results.csv")
OUT_PNG = os.path.join(RESULTS_DIR, "threshold_search_plot.png")

os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
plt.switch_backend('Agg')


def load_model_and_data() -> Tuple[MobileNetV3_CHAM_EarlyExit, torch.utils.data.DataLoader, List[str], List[int]]:
    # 加载最佳检查点（包含 class_names）
    if not os.path.exists(BEST_CHECKPOINT):
        raise FileNotFoundError(f"未找到最佳检查点: {BEST_CHECKPOINT}")
    ckpt = torch.load(BEST_CHECKPOINT, map_location='cpu')
    class_names = ckpt.get('class_names', None)
    if class_names is None:
        raise RuntimeError("best_checkpoint.pth 中未保存 class_names，无法确保类别映射一致。请重新保存或提供类名列表。")

    # 构建数据集并对齐类别映射
    processor = DataProcessor()
    master_df = processor.generate_metadata()
    val_df = master_df[master_df['split'] == 'val']

    # 以检查点中的 class_names 确定稳定映射
    class_map = {cls: idx for idx, cls in enumerate(class_names)}
    val_set = PlantDataset(val_df, mode='val')
    val_set.class_map = class_map

    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=128,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # 初始化模型
    # 从训练脚本的 EXIT_BLOCKS 推断默认值；支持从 ckpt 读取（若保存过）
    exit_blocks = ckpt.get('exit_blocks', [2, 5, 8])
    model = MobileNetV3_CHAM_EarlyExit(num_classes=len(class_names), exit_blocks=exit_blocks)
    
    # 兼容不同的权重键名与结构微调：先严格加载，失败则回退为非严格并打印差异
    state_dict = None
    for k in ['model_state', 'model', 'state_dict']:
        if k in ckpt:
            state_dict = ckpt[k]
            break
    if state_dict is None:
        state_dict = ckpt  # 兜底：假定整个ckpt就是state_dict
    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"=> 校准脚本加载权重：严格加载失败，已回退非严格。missing={len(missing)}, unexpected={len(unexpected)}")
    
    model = model.to(DEVICE)
    model.eval()

    return model, val_loader, class_names, exit_blocks


@torch.no_grad()
def collect_logits(model: MobileNetV3_CHAM_EarlyExit, loader) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """收集验证集上各出口的 logits 以及标签。"""
    logits_list: List[List[torch.Tensor]] = []  # 每个batch的list
    labels_all: List[torch.Tensor] = []
    for inputs, labels in loader:
        inputs = inputs.to(DEVICE)
        outputs_list = model(inputs)  # List[Tensor]
        if not logits_list:
            logits_list = [[] for _ in range(len(outputs_list))]
        for i, out in enumerate(outputs_list):
            logits_list[i].append(out.detach().cpu())
        labels_all.append(labels.detach().cpu())
    # 拼接
    logits_cat = [torch.cat(parts, dim=0) for parts in logits_list]
    labels_cat = torch.cat(labels_all, dim=0)
    return logits_cat, labels_cat


class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_t = nn.Parameter(torch.zeros(1))  # T=exp(log_t)，初始 T=1

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        T = torch.exp(self.log_t)
        return logits / T


def calibrate_temperatures(logits_per_exit: List[torch.Tensor], labels: torch.Tensor, steps: int = 300, lr: float = 0.05) -> List[float]:
    """对每个出口独立做温度缩放，最小化 NLL。返回每个出口的温度 T。"""
    temps: List[float] = []
    criterion = nn.CrossEntropyLoss()
    for i, logits in enumerate(logits_per_exit):
        scaler = TemperatureScaler()
        optimizer = torch.optim.Adam([scaler.log_t], lr=lr)
        for _ in range(steps):
            optimizer.zero_grad()
            logits_scaled = scaler(logits)
            loss = criterion(logits_scaled, labels)
            loss.backward()
            optimizer.step()
        T = float(torch.exp(scaler.log_t).item())
        temps.append(T)
    return temps


def apply_temperature(logits_per_exit: List[torch.Tensor], temps: List[float]) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    for logit, T in zip(logits_per_exit, temps):
        out.append(logit / T)
    return out


def simulate_dynamic_inference(probs_per_exit: List[torch.Tensor], labels: torch.Tensor, thresholds: List[float]) -> Tuple[float, List[float], float]:
    """
    根据阈值进行早退模拟。
    返回: (动态准确率, 各出口退出占比, 平均退出索引[1..N])
    """
    num_exits = len(probs_per_exit)
    total = labels.size(0)
    correct = 0
    exit_counts = [0] * num_exits
    exit_sum = 0.0

    for b in range(total):
        exited = False
        for i in range(num_exits - 1):
            conf, pred = torch.max(probs_per_exit[i][b], dim=-1)
            if conf.item() >= thresholds[i]:
                correct += int(pred.item() == labels[b].item())
                exit_counts[i] += 1
                exit_sum += (i + 1)
                exited = True
                break
        if not exited:
            pred_final = torch.argmax(probs_per_exit[-1][b]).item()
            correct += int(pred_final == labels[b].item())
            exit_counts[-1] += 1
            exit_sum += num_exits
    acc = correct / total
    ratios = [c / total for c in exit_counts]
    avg_exit_index = exit_sum / total
    return acc, ratios, avg_exit_index


def greedy_threshold_search(probs_per_exit: List[torch.Tensor], labels: torch.Tensor, init_thresholds: List[float] = None) -> Tuple[List[float], Dict]:
    """坐标下降式贪心搜索每个出口的阈值，目标最大化动态准确率。"""
    num_exits = len(probs_per_exit)
    if init_thresholds is None:
        thresholds = [0.9] * (num_exits - 1)
    else:
        thresholds = init_thresholds[:]

    candidates = np.linspace(0.50, 0.99, 11)
    history = []  # 记录每次评估的组合与指标

    best_acc, best_ratios, best_avg = simulate_dynamic_inference(probs_per_exit, labels, thresholds)
    history.append({
        't': thresholds[:], 'acc': best_acc, 'avg_exit': best_avg, 'ratios': best_ratios
    })

    for _ in range(2):  # 做两轮坐标下降
        improved = False
        for i in range(num_exits - 1):
            local_best = (best_acc, thresholds[i])
            for t in candidates:
                trial = thresholds[:]
                trial[i] = float(t)
                acc, ratios, avg_exit = simulate_dynamic_inference(probs_per_exit, labels, trial)
                history.append({'t': trial[:], 'acc': acc, 'avg_exit': avg_exit, 'ratios': ratios})
                if acc > local_best[0] + 1e-6:
                    local_best = (acc, float(t), ratios, avg_exit)
            if local_best[0] > best_acc + 1e-6:
                best_acc = local_best[0]
                thresholds[i] = float(local_best[1])
                best_ratios = local_best[2]
                best_avg = local_best[3]
                improved = True
        if not improved:
            break

    # 最终评估
    best_acc, best_ratios, best_avg = simulate_dynamic_inference(probs_per_exit, labels, thresholds)
    history.append({
        't': thresholds[:], 'acc': best_acc, 'avg_exit': best_avg, 'ratios': best_ratios
    })

    # 汇总结果
    summary = {
        'best_thresholds': thresholds,
        'best_dynamic_acc': best_acc,
        'best_exit_ratios': best_ratios,
        'best_avg_exit_index': best_avg,
        'history': history
    }
    return thresholds, summary


def save_history_csv(history: List[Dict]):
    # 扁平化写入 CSV
    rows = []
    for rec in history:
        t = rec['t']
        acc = rec['acc']
        avg_exit = rec['avg_exit']
        ratios = rec['ratios']
        row = [acc, avg_exit] + t + ratios
        rows.append(row)
    # 动态列头
    if len(history) == 0:
        return
    num_exits = len(history[-1]['ratios'])
    headers = ["dynamic_acc", "avg_exit_index"]
    headers += [f"th_exit{i+1}" for i in range(num_exits - 1)]
    headers += [f"ratio_exit{i+1}" for i in range(num_exits)]

    arr = np.array(rows, dtype=float)
    np.savetxt(OUT_CSV, arr, delimiter=',', header=','.join(headers), comments='')


def save_plot(history: List[Dict], best_summary: Dict):
    # 绘制动态准确率 vs 平均退出索引散点图
    xs = [rec['avg_exit'] for rec in history]
    ys = [rec['acc'] for rec in history]
    plt.figure(figsize=(8,6))
    plt.scatter(xs, ys, s=20, alpha=0.5, label='Candidates')
    plt.scatter([best_summary['best_avg_exit_index']], [best_summary['best_dynamic_acc']],
                color='red', s=80, label='Best')
    plt.xlabel('Average Exit Index (lower is earlier)')
    plt.ylabel('Dynamic Accuracy')
    plt.title('Threshold Search (Temperature Scaled)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PNG)


def main():
    model, val_loader, class_names, exit_blocks = load_model_and_data()

    # 收集 logits
    logits_per_exit, labels = collect_logits(model, val_loader)

    # 温度缩放（按出口独立校准）
    temps = calibrate_temperatures(logits_per_exit, labels, steps=300, lr=0.05)
    print(f"Calibrated temperatures per exit: {temps}")

    # 计算校准后的概率
    logits_scaled = apply_temperature(logits_per_exit, temps)
    probs_per_exit = [torch.softmax(l, dim=-1) for l in logits_scaled]

    # 阈值贪心搜索
    best_th, summary = greedy_threshold_search(probs_per_exit, labels, init_thresholds=None)
    print("Best thresholds (for early exits):", best_th)
    print("Best dynamic accuracy:", summary['best_dynamic_acc'])
    print("Best exit ratios:", [round(r,3) for r in summary['best_exit_ratios']])
    print("Best avg exit index:", round(summary['best_avg_exit_index'],3))

    # 保存对比表与图
    save_history_csv(summary['history'])
    save_plot(summary['history'], summary)

    # 另存一份文本摘要
    with open(os.path.join(RESULTS_DIR, 'threshold_search_summary.txt'), 'w') as f:
        f.write(f"exit_blocks: {exit_blocks}\n")
        f.write(f"temperatures: {temps}\n")
        f.write(f"best_thresholds: {best_th}\n")
        f.write(f"best_dynamic_acc: {summary['best_dynamic_acc']:.6f}\n")
        f.write(f"best_exit_ratios: {[round(r,6) for r in summary['best_exit_ratios']]}\n")
        f.write(f"best_avg_exit_index: {summary['best_avg_exit_index']:.6f}\n")


if __name__ == '__main__':
    main()