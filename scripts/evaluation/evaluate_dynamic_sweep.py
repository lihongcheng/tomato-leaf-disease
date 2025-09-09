import os
import time
import argparse
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from calibrate_thresholds_and_search import (
    load_model_and_data,
    collect_logits,
    apply_temperature,
    simulate_dynamic_inference,
)
from train_mobilenetv3_cham_earlyexit import compute_flops_costs

# 路径配置
RESULTS_DIR = "./results/MobileNetV3_CHAM_EarlyExit"
SUMMARY_TXT = os.path.join(RESULTS_DIR, "threshold_search_summary.txt")
SWEEP_CSV = os.path.join(RESULTS_DIR, "dynamic_threshold_sweep.csv")
PARETO_LAT_SVG = os.path.join(RESULTS_DIR, "Fig_dynamic_accuracy_latency_pareto.svg")
PARETO_FLOPS_SVG = os.path.join(RESULTS_DIR, "Fig_dynamic_accuracy_flops_pareto.svg")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
INPUT_RES = 224


def load_thresholds_and_temps(summary_path: str) -> Tuple[List[int], List[float], List[float]]:
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"未找到阈值与温度摘要文件: {summary_path}，请先运行 calibrate_thresholds_and_search.py")
    exit_blocks, temps, best_th = None, None, None
    with open(summary_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('exit_blocks:'):
                s = line.split(':', 1)[1].strip()
                exit_blocks = eval(s)
            elif line.startswith('temperatures:'):
                s = line.split(':', 1)[1].strip()
                temps = eval(s)
            elif line.startswith('best_thresholds:'):
                s = line.split(':', 1)[1].strip()
                best_th = eval(s)
    if exit_blocks is None or temps is None:
        raise RuntimeError("阈值摘要文件缺少必要字段(exit_blocks/temperatures)。")
    return exit_blocks, temps, best_th


class ExitPath(nn.Module):
    def __init__(self, orig: nn.Module, exit_idx: int):
        super().__init__()
        import copy
        self.features = nn.Sequential(*(copy.deepcopy(m) for m in list(orig.features.children())[:exit_idx + 1]))
        self.cham_layers = nn.ModuleDict({k: copy.deepcopy(v) for k, v in orig.cham_layers.items() if int(k) <= exit_idx})
        self.exit_head = copy.deepcopy(orig.exit_heads[str(exit_idx)])

    def forward(self, x):
        for i, m in enumerate(self.features):
            x = m(x)
            if str(i) in self.cham_layers:
                x = self.cham_layers[str(i)](x)
        return self.exit_head(x)


class FullPath(nn.Module):
    def __init__(self, orig: nn.Module):
        super().__init__()
        import copy
        self.features = copy.deepcopy(orig.features)
        self.cham_layers = nn.ModuleDict({k: copy.deepcopy(v) for k, v in orig.cham_layers.items()})
        self.final_pool = copy.deepcopy(orig.final_pool)
        self.classifier = copy.deepcopy(orig.classifier)

    def forward(self, x):
        for i, m in enumerate(self.features):
            x = m(x)
            if str(i) in self.cham_layers:
                x = self.cham_layers[str(i)](x)
        xf = self.final_pool(x).flatten(1)
        return self.classifier(xf)


def benchmark_latency_per_path(exit_blocks: List[int], model: nn.Module, device: torch.device, input_res: int = 224,
                               warmup: int = 5, iters: int = 50) -> Tuple[List[float], float]:
    """返回：每个早退路径单样本平均延迟(ms)列表、全路径单样本平均延迟(ms)"""
    model.eval()
    x = torch.randn(1, 3, input_res, input_res, device=device)
    # 构建路径包装器
    wrappers = [ExitPath(model, idx).to(device).eval() for idx in exit_blocks]
    final_wrapper = FullPath(model).to(device).eval()

    def time_module(m: nn.Module) -> float:
        # 预热
        with torch.no_grad():
            for _ in range(warmup):
                _ = m(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(iters):
                _ = m(x)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        return (t1 - t0) * 1000.0 / iters  # ms/样本

    lat_exits_ms = [time_module(w) for w in wrappers]
    lat_full_ms = time_module(final_wrapper)
    return lat_exits_ms, lat_full_ms


def compute_expected_costs(exit_ratios: List[float], flops_exits_g: List[float], flops_full_g: float,
                           lat_exits_ms: List[float], lat_full_ms: float) -> Tuple[float, float]:
    num_exits = len(exit_ratios)
    dyn_flops_g = 0.0
    for i in range(num_exits - 1):
        dyn_flops_g += exit_ratios[i] * flops_exits_g[i]
    dyn_flops_g += exit_ratios[-1] * flops_full_g

    avg_latency_ms = 0.0
    for i in range(num_exits - 1):
        avg_latency_ms += exit_ratios[i] * lat_exits_ms[i]
    avg_latency_ms += exit_ratios[-1] * lat_full_ms
    return float(dyn_flops_g), float(avg_latency_ms)


def pareto_front(points: List[Tuple[float, float]]) -> List[int]:
    """返回非支配点的索引，目标：最大化accuracy，最小化cost(第二维)。"""
    idxs = []
    for i, (acc_i, cost_i) in enumerate(points):
        dominated = False
        for j, (acc_j, cost_j) in enumerate(points):
            if j == i:
                continue
            # j dominates i if acc_j >= acc_i and cost_j <= cost_i, with at least one strict
            if (acc_j >= acc_i and cost_j <= cost_i) and (acc_j > acc_i or cost_j < cost_i):
                dominated = True
                break
        if not dominated:
            idxs.append(i)
    return idxs


def main():
    parser = argparse.ArgumentParser(description="多阈值扫掠，输出精度-时延/FLOPs帕累托散点图")
    parser.add_argument('--t-min', type=float, default=0.5, help='阈值最小值')
    parser.add_argument('--t-max', type=float, default=0.99, help='阈值最大值')
    parser.add_argument('--t-num', type=int, default=11, help='阈值采样点数')
    parser.add_argument('--non-uniform', action='store_true', help='是否对每个早退出口使用不同阈值（默认统一阈值）')
    parser.add_argument('--iters', type=int, default=50, help='延迟基准测试迭代次数')
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1) 加载模型与数据
    model, val_loader, class_names, exit_blocks_model = load_model_and_data()

    # 2) 读取温度与参考阈值
    exit_blocks_txt, temps, best_th = load_thresholds_and_temps(SUMMARY_TXT)
    if exit_blocks_txt != exit_blocks_model:
        print(f"[警告] 摘要文件中的 exit_blocks={exit_blocks_txt} 与当前模型的 exit_blocks={exit_blocks_model} 不一致，继续使用模型内的设置。")

    # 3) 收集各出口 logits 并计算概率（应用温度缩放）
    logits_per_exit, labels = collect_logits(model, val_loader)
    logits_scaled = apply_temperature(logits_per_exit, temps)
    probs_per_exit = [torch.softmax(l, dim=-1) for l in logits_scaled]

    # 3.5) 计算最终出口的静态准确率（作为基线）
    with torch.no_grad():
        final_logits = logits_scaled[-1]
        final_pred = final_logits.argmax(dim=1)
        static_final_acc = (final_pred == labels).float().mean().item()

    # 4) 预计算路径FLOPs和延迟
    flops_exits_g, flops_full_g = compute_flops_costs(model, exit_blocks_model, input_res=INPUT_RES)
    lat_exits_ms, lat_full_ms = benchmark_latency_per_path(exit_blocks_model, model, DEVICE, input_res=INPUT_RES, iters=args.iters)

    # 5) 阈值扫掠（默认统一阈值）
    thresholds_list: List[List[float]] = []
    grid = np.linspace(args.t_min, args.t_max, args.t_num).tolist()
    if not args.non_uniform:
        for t in grid:
            thresholds_list.append([float(t) for _ in exit_blocks_model])
    else:
        # 简单的笛卡尔积（注意组合数量会迅速膨胀）
        import itertools
        for comb in itertools.product(grid, repeat=len(exit_blocks_model)):
            thresholds_list.append([float(x) for x in comb])

    # 6) 模拟并统计
    records: List[Dict] = []
    for thr in thresholds_list:
        acc_dyn, exit_ratios, avg_exit_index = simulate_dynamic_inference(probs_per_exit, labels, thr)
        dyn_flops_g, avg_latency_ms = compute_expected_costs(exit_ratios, flops_exits_g, flops_full_g, lat_exits_ms, lat_full_ms)
        records.append({
            'thresholds': thr,
            'accuracy': float(acc_dyn),
            'avg_exit_index': float(avg_exit_index),
            'dynamic_flops_g': float(dyn_flops_g),
            'avg_latency_ms': float(avg_latency_ms),
            'exit_ratios': [float(r) for r in exit_ratios],
        })

    # 7) 保存CSV
    with open(SWEEP_CSV, 'w') as f:
        f.write('thresholds,accuracy,avg_exit_index,dynamic_flops_g,avg_latency_ms,exit_ratios\n')
        for r in records:
            f.write(f"{r['thresholds']},{r['accuracy']:.6f},{r['avg_exit_index']:.6f},{r['dynamic_flops_g']:.6f},{r['avg_latency_ms']:.6f},{r['exit_ratios']}\n")
    print(f"[保存] 阈值扫掠结果: {SWEEP_CSV}")

    # 8) 绘制帕累托散点图（Acc vs Latency, Acc vs FLOPs）
    accs = np.array([r['accuracy'] for r in records])
    lats = np.array([r['avg_latency_ms'] for r in records])
    flops = np.array([r['dynamic_flops_g'] for r in records])

    # 计算并高亮帕累托前沿
    pareto_idx_lat = pareto_front(list(zip(accs, lats)))
    pareto_idx_flops = pareto_front(list(zip(accs, flops)))

    # 静态基线（最终出口）
    static_latency = lat_full_ms
    static_flops = flops_full_g

    # Acc-Latency
    plt.figure(figsize=(7, 5))
    plt.scatter(lats, accs, s=24, c='#1f77b4', alpha=0.6, label='Dynamic (sweep)')
    if len(pareto_idx_lat) > 0:
        plt.scatter(lats[pareto_idx_lat], accs[pareto_idx_lat], s=36, c='#d62728', label='Pareto front')
    plt.scatter([static_latency], [static_final_acc], marker='x', c='k', s=70, label='Static final')
    plt.xlabel('Estimated average latency (ms)')
    plt.ylabel('Dynamic accuracy (val)')
    plt.title('Accuracy vs Latency (Dynamic early-exit sweep)')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PARETO_LAT_SVG, format='svg', bbox_inches='tight')
    plt.close()
    print(f"[保存] 帕累托图: {PARETO_LAT_SVG}")

    # Acc-FLOPs
    plt.figure(figsize=(7, 5))
    plt.scatter(flops, accs, s=24, c='#2ca02c', alpha=0.6, label='Dynamic (sweep)')
    if len(pareto_idx_flops) > 0:
        plt.scatter(flops[pareto_idx_flops], accs[pareto_idx_flops], s=36, c='#d62728', label='Pareto front')
    plt.scatter([static_flops], [static_final_acc], marker='x', c='k', s=70, label='Static final')
    plt.xlabel('Estimated dynamic FLOPs (GFLOPs)')
    plt.ylabel('Dynamic accuracy (val)')
    plt.title('Accuracy vs FLOPs (Dynamic early-exit sweep)')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PARETO_FLOPS_SVG, format='svg', bbox_inches='tight')
    plt.close()
    print(f"[保存] 帕累托图: {PARETO_FLOPS_SVG}")


if __name__ == '__main__':
    main()