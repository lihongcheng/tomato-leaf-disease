import os
import time
import copy
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn

# 复用已有工具与流程
from calibrate_thresholds_and_search import (
    load_model_and_data,
    collect_logits,
    apply_temperature,
    simulate_dynamic_inference,
)
from train_mobilenetv3_cham_earlyexit import compute_flops_costs


RESULTS_DIR = "./results/MobileNetV3_CHAM_EarlyExit"
CHECKPOINT_DIR = "./checkpoints/MobileNetV3_CHAM_EarlyExit"
BEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_checkpoint.pth")
SUMMARY_TXT = os.path.join(RESULTS_DIR, "threshold_search_summary.txt")
DYNAMIC_EVAL_TXT = os.path.join(RESULTS_DIR, "dynamic_eval_summary.txt")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
INPUT_RES = 224  # 与训练/评估保持一致


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
    if exit_blocks is None or temps is None or best_th is None:
        raise RuntimeError("阈值摘要文件缺少必要字段(exit_blocks/temperatures/best_thresholds)。")
    return exit_blocks, temps, best_th


class ExitPath(nn.Module):
    def __init__(self, orig: nn.Module, exit_idx: int):
        super().__init__()
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


def main():
    # 1) 加载模型与数据
    model, val_loader, class_names, exit_blocks_model = load_model_and_data()

    # 2) 读取温度与最佳阈值
    exit_blocks_txt, temps, best_th = load_thresholds_and_temps(SUMMARY_TXT)
    if exit_blocks_txt != exit_blocks_model:
        print(f"[警告] 摘要文件中的 exit_blocks={exit_blocks_txt} 与当前模型的 exit_blocks={exit_blocks_model} 不一致，继续使用模型内的设置。")
    num_exits = len(exit_blocks_model) + 1  # 包含最终出口

    # 3) 收集各出口 logits 并计算概率（应用温度缩放）
    logits_per_exit, labels = collect_logits(model, val_loader)
    logits_scaled = apply_temperature(logits_per_exit, temps)
    probs_per_exit = [torch.softmax(l, dim=-1) for l in logits_scaled]

    # 4) 根据阈值模拟动态推理，得到退出占比与平均退出索引
    acc_dyn, exit_ratios, avg_exit_index = simulate_dynamic_inference(probs_per_exit, labels, best_th)

    # 5) FLOPs：使用已有函数估算各出口与全路径GFLOPs，并计算期望动态FLOPs
    flops_exits_g, flops_full_g = compute_flops_costs(model, exit_blocks_model, input_res=INPUT_RES)
    # exit_ratios 对应 [每个早退, 最终]
    dyn_flops_g = 0.0
    for i in range(num_exits - 1):
        dyn_flops_g += exit_ratios[i] * flops_exits_g[i]
    dyn_flops_g += exit_ratios[-1] * flops_full_g

    # 6) 延迟：对每条路径做微基准测试，估计单样本延迟，然后按退出占比加权
    lat_exits_ms, lat_full_ms = benchmark_latency_per_path(exit_blocks_model, model, DEVICE, input_res=INPUT_RES)
    avg_latency_ms = 0.0
    for i in range(num_exits - 1):
        avg_latency_ms += exit_ratios[i] * lat_exits_ms[i]
    avg_latency_ms += exit_ratios[-1] * lat_full_ms

    # 7) 打印与保存结果
    print("=== Dynamic Early-Exit Evaluation (Simulation) ===")
    print(f"Num exits (incl. final): {num_exits}")
    print(f"Best thresholds: {best_th}")
    print(f"Exit ratios: {[round(r, 4) for r in exit_ratios]}")
    print(f"Average exit index: {avg_exit_index:.4f}")
    print(f"Estimated dynamic FLOPs (GFLOPs): {dyn_flops_g:.4f}")
    print(f"Estimated average latency (ms/image): {avg_latency_ms:.3f}")
    print("Per-path details:")
    for i, idx in enumerate(exit_blocks_model):
        print(f"  - Exit@block{idx}: FLOPs={flops_exits_g[i]:.4f} GFLOPs, Latency={lat_exits_ms[i]:.3f} ms")
    print(f"  - Final path: FLOPs={flops_full_g:.4f} GFLOPs, Latency={lat_full_ms:.3f} ms")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(DYNAMIC_EVAL_TXT, 'w') as f:
        f.write("Dynamic Early-Exit Evaluation (Simulation)\n")
        f.write(f"best_thresholds: {best_th}\n")
        f.write(f"exit_ratios: {[float(r) for r in exit_ratios]}\n")
        f.write(f"avg_exit_index: {float(avg_exit_index):.6f}\n")
        f.write(f"dynamic_flops_g: {float(dyn_flops_g):.6f}\n")
        f.write(f"avg_latency_ms: {float(avg_latency_ms):.6f}\n")
        f.write("per_path:\n")
        for i, idx in enumerate(exit_blocks_model):
            f.write(f"  exit_block_{idx}: flops_g={float(flops_exits_g[i]):.6f}, latency_ms={float(lat_exits_ms[i]):.6f}\n")
        f.write(f"  final: flops_g={float(flops_full_g):.6f}, latency_ms={float(lat_full_ms):.6f}\n")


if __name__ == "__main__":
    main()