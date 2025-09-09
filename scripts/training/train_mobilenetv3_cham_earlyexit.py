import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from data_process import DataProcessor, PlantDataset
from contextlib import nullcontext
# 模型延迟导入，见 main() 中的 try/except
import os
import numpy as np
import torchvision.models as tv_models
from ptflops import get_model_complexity_info
import copy

# 配置参数
plt.switch_backend('Agg')
RESULTS_DIR = "./results/MobileNetV3_CHAM_EarlyExit" # 修改结果目录
FIG_PATH = os.path.join(RESULTS_DIR, "training_curve.png")
LOG_PATH = os.path.join(RESULTS_DIR, "training_log.csv")
CHECKPOINT_DIR = "./checkpoints/MobileNetV3_CHAM_EarlyExit" # 修改检查点目录
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 检查点配置
RESUME_CHECKPOINT = True
LATEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")
BEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_checkpoint.pth")

# 早停配置
EARLY_STOP_PATIENCE = 25
MIN_DELTA = 0.001

# 硬件配置
BATCH_SIZE = 128
NUM_WORKERS = 14
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS = 100

# --- 早期退出特定配置 ---
# 定义早期退出分支的位置（例如：对应MobileNetV3_small的某些bottleneck索引）
EXIT_BLOCKS = [2, 5, 8] # 示例值，需根据你的实际模型结构确定
# 每个退出分支的损失权重 [exit1, exit2, exit3, final_exit]
# 使用衰减权重：让网络前期学习所有分支，后期专注主输出
ALPHA = [0.3, 0.2, 0.1, 1.0]
ALPHA_DECAY_FACTOR = 0.85
# 动态推理阈值（与早退分支一一对应），用于验证阶段的动态退出模拟
THRESHOLDS = [0.7 for _ in EXIT_BLOCKS]  # 将初始阈值从0.9下调至0.7，避免前期永不早退
# 自动阈值校准：在验证集上周期性搜索更优阈值（提高动态准确率/降低平均计算量）
AUTO_CALIBRATE_THRESHOLDS = True
CALIBRATION_START_EPOCH = int(EPOCHS * 0.4)
CALIBRATION_FREQ = 5  # 每N个epoch校准一次
# 结构化剪枝友好：对BN权重施加L1稀疏正则，便于后续按通道剪枝
PRUNE_REG_ENABLE = True
PRUNE_REG_LAMBDA = 1e-5

# === Priority-A: KD/EMA 与两阶段训练超参 ===
# 阶段1训练占比（0~1），前期多出口训练，后期聚焦最终出口
STAGE1_RATIO = 0.3
# 知识蒸馏开关与超参（以最终出口为教师，蒸馏早退分支）
KD_ENABLE = True
KD_TEMPERATURE = 2.0
KD_LAMBDA_STAGE1 = 0.5
KD_LAMBDA_STAGE2 = 0.2
# EMA权重滑动平均的衰减系数
EMA_DECAY = 0.999
# 单分支 MobileNetV3 的 best 检查点路径（用于可选初始化）
INIT_FROM_SINGLE_BRANCH = "./checkpoints/MobileNetV3/best_checkpoint.pth"
# 模型瘦身：宽度系数（<1.0 可显著减少参数与FLOPs）
WIDTH_MULT = 1.0
# 额外教师蒸馏（使用单分支最佳模型作为外部教师）
EXT_TEACHER_ENABLE = False
EXT_TEACHER_CKPT = "./checkpoints/MobileNetV3/best_checkpoint.pth"
KD_EXT_LAMBDA_STAGE1 = 0.7
KD_EXT_LAMBDA_STAGE2 = 0.3
KD_EXT_TO_EARLY = True

class TrainingVisualizer:
    """训练过程可视化器（扩展：记录多个出口的准确率、动态准确率与退出占比）"""
    def __init__(self, num_exits):
        self.train_loss = []
        self.val_loss = []
        self.val_acc = []              # 最终出口准确率
        # 记录每个退出分支的验证准确率
        self.val_acc_exits = [[] for _ in range(num_exits)]
        # 动态推理：整体准确率与各出口退出占比
        self.dynamic_acc = []          # 动态策略下的整体准确率
        self.exit_ratio_history = [[] for _ in range(num_exits)]  # 每个出口的退出占比曲线
        self.dynamic_flops_g = []      # 记录期望动态FLOPs（G）

        self.fig, self.ax = plt.subplots(2, 1, figsize=(10, 10))
        self.num_exits = num_exits
        
    def update(self, epoch, t_loss, v_loss, v_acc_final, v_acc_exits, v_acc_dynamic, exit_ratios, expected_dyn_flops_g):
        self.train_loss.append(t_loss)
        self.val_loss.append(v_loss)
        self.val_acc.append(v_acc_final)
        
        for i, acc in enumerate(v_acc_exits):
            self.val_acc_exits[i].append(acc)
        
        self.dynamic_acc.append(v_acc_dynamic)
        self.dynamic_flops_g.append(expected_dyn_flops_g)
        for i, r in enumerate(exit_ratios):
            self.exit_ratio_history[i].append(r)
        
        self.ax[0].clear()
        self.ax[0].plot(range(1, epoch+2), self.train_loss, label='Train Loss')
        self.ax[0].plot(range(1, epoch+2), self.val_loss, label='Val Loss')
        self.ax[0].set_title('Loss Curve')
        self.ax[0].legend()
        
        self.ax[1].clear()
        self.ax[1].plot(range(1, epoch+2), self.val_acc, 'g-', label='Val Acc (Final)')
        # 绘制每个退出分支的准确率曲线
        colors = ['r--', 'b--', 'c--', 'm--', 'y--', 'k--']
        for i in range(self.num_exits):
            color = colors[i % len(colors)]
            self.ax[1].plot(range(1, epoch+2), self.val_acc_exits[i], color, label=f'Val Acc (Exit {i+1})', alpha=0.7)
        # 动态推理整体准确率
        self.ax[1].plot(range(1, epoch+2), self.dynamic_acc, 'k-.', label='Val Acc (Dynamic)')

        self.ax[1].set_title('Accuracy Curve')
        self.ax[1].set_ylim(0.5, 1.0)
        self.ax[1].legend()
        
        self.fig.tight_layout()
        self.fig.savefig(FIG_PATH)
        # 保存日志也需要扩展
        all_data = [self.train_loss, self.val_loss, self.val_acc]
        all_data.extend(self.val_acc_exits)
        all_data.append(self.dynamic_acc)
        all_data.append(self.dynamic_flops_g)
        for i in range(self.num_exits):
            all_data.append(self.exit_ratio_history[i])
        header = 'train_loss,val_loss,val_acc_final'
        for i in range(self.num_exits):
            header += f',val_acc_exit{i+1}'
        header += ',val_acc_dynamic,dyn_flops_g'
        for i in range(self.num_exits):
            header += f',exit_ratio_exit{i+1}'
        np.savetxt(LOG_PATH, 
                  np.column_stack(all_data),
                  delimiter=',',
                  header=header,
                  comments='')

class EarlyStopper:
    """早停控制器（基于最终出口的准确率）"""
    def __init__(self):
        self.best_acc = 0.0
        self.counter = 0
        self.early_stop = False
        
    def check(self, val_acc):
        if val_acc > self.best_acc + MIN_DELTA:
            self.best_acc = val_acc
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= EARLY_STOP_PATIENCE:
                self.early_stop = True
        return self.early_stop

class LabelSmoothCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        log_preds = torch.log_softmax(logits, dim=-1)
        with torch.no_grad():
            targets = torch.zeros_like(log_preds).scatter_(
                1, targets.unsqueeze(1), (1 - self.smoothing))
            targets += self.smoothing / num_classes
        loss = -torch.sum(targets * log_preds, dim=-1).mean()
        return loss

# === Priority-A 新增：KD 与 EMA 工具 ===
def kd_kl_div(student_logits: torch.Tensor, teacher_logits: torch.Tensor, T: float) -> torch.Tensor:
    """KL(student || teacher) with temperature scaling and batch mean reduction."""
    return F.kl_div(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
        reduction='batchmean'
    ) * (T * T)

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items() if v.dtype.is_floating_point}
        self.backup = {}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.state_dict().items():
            if name in self.shadow and param.dtype.is_floating_point:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        state = model.state_dict()
        for name, val in self.shadow.items():
            if name in state:
                self.backup[name] = state[name].detach().clone()
                state[name].copy_(val)

    def restore(self, model: nn.Module):
        if not self.backup:
            return
        state = model.state_dict()
        for name, val in self.backup.items():
            if name in state:
                state[name].copy_(val)
        self.backup = {}

# 新增：按路径计算FLOPs（各早退与最终出口）
def compute_flops_costs(model: nn.Module, exit_blocks: list, input_res: int = 224):
    """
    返回：
      - flops_exits_g: 长度=len(exit_blocks)，每个早退路径的GFLOPs
      - flops_full_g: 最终全路径GFLOPs
    说明：使用深拷贝构建路径包裹器，避免模块多重注册冲突。
    """
    class ExitPath(nn.Module):
        def __init__(self, orig: nn.Module, exit_idx: int):
            super().__init__()
            # features到指定层（包含）
            self.features = nn.Sequential(*(copy.deepcopy(m) for m in list(orig.features.children())[:exit_idx+1]))
            # 仅保留<=exit_idx的CHAM
            self.cham_layers = nn.ModuleDict({k: copy.deepcopy(v) for k, v in orig.cham_layers.items() if int(k) <= exit_idx})
            # 对应的exit head
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
    # 构建包裹器
    exit_wrappers = [ExitPath(model, idx) for idx in exit_blocks]
    final_wrapper = FullPath(model)
    # 计算MACs并转FLOPs
    flops_exits_g = []
    for w in exit_wrappers:
        macs, _ = get_model_complexity_info(w, (3, input_res, input_res), as_strings=False, print_per_layer_stat=False, verbose=False)
        flops_exits_g.append(float(macs) * 2.0 / 1e9)
    macs_full, _ = get_model_complexity_info(final_wrapper, (3, input_res, input_res), as_strings=False, print_per_layer_stat=False, verbose=False)
    flops_full_g = float(macs_full) * 2.0 / 1e9
    return flops_exits_g, flops_full_g


def save_checkpoint(state, filename):
    torch.save(state, filename)
    print(f"=> 保存检查点到 {filename}")

def load_checkpoint(filename, model, optimizer=None, scheduler=None):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location='cpu')
        # 先尝试严格加载；失败则回退为非严格，兼容结构小改动
        try:
            model.load_state_dict(checkpoint['model_state'], strict=True)
        except Exception as e:
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint['model_state'], strict=False)
            print(f"=> 模型权重严格加载失败，已回退为非严格加载。missing={len(missing_keys)}, unexpected={len(unexpected_keys)}")
        if optimizer is not None and 'optimizer_state' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state'])
            except Exception:
                print("=> 优化器状态与当前设置不匹配，已跳过恢复。")
        if scheduler is not None and 'scheduler_state' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state'])
            except Exception:
                print("=> 调度器状态与当前设置不匹配，已跳过恢复。")
        print(f"=> 从 {filename} 加载检查点 (epoch {checkpoint['epoch']})")
        return checkpoint
    else:
        print(f"=> 未找到检查点 {filename}")
        return None

# 在EMA评估权重下，基于验证集贪心校准多出口阈值（小网格搜索）
def calibrate_thresholds(model: nn.Module, val_loader, num_exits: int, base_thresholds: list, device: torch.device):
    thresholds = base_thresholds.copy()

    def eval_with(thr_list):
        total, correct = 0, 0
        exit_counts = [0] * num_exits
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs_list = model(inputs)
                batch_size = labels.size(0)
                total += batch_size
                for b in range(batch_size):
                    exited = False
                    for i in range(num_exits - 1):
                        probs = torch.softmax(outputs_list[i][b], dim=-1)
                        conf, pred = torch.max(probs, dim=-1)
                        if conf.item() >= thr_list[i]:
                            if pred.item() == labels[b].item():
                                correct += 1
                            exit_counts[i] += 1
                            exited = True
                            break
                    if not exited:
                        probs_final = torch.softmax(outputs_list[-1][b], dim=-1)
                        pred_final = torch.argmax(probs_final).item()
                        if pred_final == labels[b].item():
                            correct += 1
                        exit_counts[-1] += 1
        dyn_acc = correct / max(1, total)
        ratios = [c / max(1, total) for c in exit_counts]
        return dyn_acc, ratios

    # 候选阈值（轻量搜索，控制计算开销）
    candidates = [0.7, 0.8, 0.85, 0.9, 0.95]
    # 贪心逐出口校准（固定之前分支阈值）
    for i in range(num_exits - 1):
        best_t, best_acc = thresholds[i], -1.0
        for t in candidates:
            trial = thresholds.copy()
            trial[i] = t
            acc, _ = eval_with(trial)
            if acc > best_acc:
                best_acc = acc
                best_t = t
        thresholds[i] = best_t
    return thresholds

def main():
    processor = DataProcessor()

    # 先构建元数据与数据集，得到稳定的类别映射
    master_df = processor.generate_metadata()
    class_names = sorted(master_df['class'].unique())
    class_map = {cls: idx for idx, cls in enumerate(class_names)}

    train_df = master_df[master_df['split'] == 'train']
    val_df = master_df[master_df['split'] == 'val']

    train_set = PlantDataset(train_df, mode='train')
    val_set = PlantDataset(val_df, mode='val')
    # 强制对齐两端的类映射，避免不同split导致的label不一致
    train_set.class_map = class_map
    val_set.class_map = class_map

    # 延迟导入模型定义，给出更清晰的报错信息
    try:
        from model_def import MobileNetV3_CHAM_EarlyExit
    except Exception as e:
        raise ImportError(
            "未找到 model_def.MobileNetV3_CHAM_EarlyExit，请实现该模型类（包含早期退出分支与CHAM模块），或调整导入路径。"
        ) from e

    # 初始化模型（基于稳定的类别数和早退配置）
    model = MobileNetV3_CHAM_EarlyExit(num_classes=len(class_names), exit_blocks=EXIT_BLOCKS, width_mult=WIDTH_MULT)
    model = model.to(DEVICE)
    num_exits = len(EXIT_BLOCKS) + 1 # +1 for the final exit
    # 运行时阈值（可被校准与断点恢复）
    thresholds_rt = THRESHOLDS.copy()
    # 打印参数量（M）
    total_params_m = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model params: {total_params_m:.2f}M | width_mult={WIDTH_MULT}")
    # 新增：静态FLOPs统计（各早退路径与最终路径）
    try:
        flops_exits_g, flops_full_g = compute_flops_costs(model.cpu(), EXIT_BLOCKS, input_res=224)
        # 将模型搬回设备
        model = model.to(DEVICE)
        print(f"FLOPs(Full): {flops_full_g:.4f} G | FLOPs(Exits): {[round(f,4) for f in flops_exits_g]}")
    except Exception as e:
        print(f"FLOPs计算失败：{e}")
        flops_exits_g = [float('nan')] * len(EXIT_BLOCKS)
        flops_full_g = float('nan')

    # Priority-A: EMA 管理器
    ema = ModelEMA(model, decay=EMA_DECAY)
    
    # 预收集BN权重引用（稀疏正则目标）
    bn_weights = [m.weight for m in model.modules() if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d))]
    total_bn = sum(w.numel() for w in bn_weights) if bn_weights else 1

    # 可选：从单分支检查点初始化（仅当不从latest恢复时）
    if (not (RESUME_CHECKPOINT and os.path.exists(LATEST_CHECKPOINT))) and INIT_FROM_SINGLE_BRANCH and os.path.exists(INIT_FROM_SINGLE_BRANCH):
        try:
            sb_ckpt = torch.load(INIT_FROM_SINGLE_BRANCH, map_location='cpu')
            state = sb_ckpt.get('model_state') or sb_ckpt.get('state_dict') or sb_ckpt.get('model')
            if state is not None:
                missing, unexpected = model.load_state_dict(state, strict=False)
                print(f"=> 从单分支初始化完成：missing={len(missing)}, unexpected={len(unexpected)}")
                # 初始化EMA为当前权重
                ema = ModelEMA(model, decay=EMA_DECAY)
            else:
                print("=> 单分支检查点中未找到可用的权重键（model_state/state_dict/model）。")
        except Exception as e:
            print(f"=> 单分支初始化失败：{e}")

    # 外部教师：单分支最佳模型（可选）
    ext_teacher = None
    if EXT_TEACHER_ENABLE and os.path.exists(EXT_TEACHER_CKPT):
        try:
            ext_teacher = tv_models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        except Exception:
            ext_teacher = tv_models.mobilenet_v3_small(weights=None)
        # 适配类别数
        try:
            ext_teacher.classifier[3] = nn.Linear(ext_teacher.classifier[3].in_features, len(class_names))
        except Exception:
            # 兼容不同torchvision版本：最后一层通常是[-1]
            in_features = ext_teacher.classifier[-1].in_features
            ext_teacher.classifier[-1] = nn.Linear(in_features, len(class_names))
        # 加载权重（严格检查分类头是否成功加载，否则禁用外部教师）
        try:
            t_ckpt = torch.load(EXT_TEACHER_CKPT, map_location='cpu')
            t_state = t_ckpt.get('model_state') or t_ckpt.get('state_dict') or t_ckpt.get('model')
            if t_state is not None:
                load_res = ext_teacher.load_state_dict(t_state, strict=False)
                missing_keys = set(getattr(load_res, 'missing_keys', []) or [])
                # 如分类头参数缺失，说明类别数/结构不匹配，使用该教师会产生随机无效引导 -> 禁用
                classifier_missing = any(k.startswith('classifier') for k in missing_keys)
                if classifier_missing:
                    print("=> 外部教师分类头与数据集不匹配，已禁用外部教师KD（避免随机教师干扰训练）")
                    ext_teacher = None
                else:
                    print("=> 外部教师权重加载完成（单分支best）")
            else:
                print("=> 外部教师检查点缺少可用权重键，已禁用外部教师KD")
                ext_teacher = None
        except Exception as e:
            print(f"=> 外部教师加载失败：{e}")
            ext_teacher = None
        if ext_teacher is not None:
            ext_teacher.to(DEVICE)
            ext_teacher.eval()
            for p in ext_teacher.parameters():
                p.requires_grad = False
    else:
        if EXT_TEACHER_ENABLE:
            print(f"=> 未找到外部教师检查点：{EXT_TEACHER_CKPT}，已禁用外部教师KD")

    visualizer = TrainingVisualizer(num_exits=num_exits)
    early_stopper = EarlyStopper()

    # 损失函数和优化器（先构建优化器与 scaler，调度器在 DataLoader 创建后初始化）
    criterion = LabelSmoothCrossEntropy(smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=2e-5)
    scaler = GradScaler(enabled=(DEVICE.type == 'cuda'))

    # 初始化损失权重（阶段1）与KD系数
    current_alpha = ALPHA.copy()
    STAGE1_EPOCHS = max(1, int(EPOCHS * STAGE1_RATIO))
    kd_lambda = KD_LAMBDA_STAGE1 if KD_ENABLE else 0.0
    kd_ext_lambda = KD_EXT_LAMBDA_STAGE1 if (EXT_TEACHER_ENABLE and (ext_teacher is not None)) else 0.0
    stage2_started = False

    # 数据加载器（移除 WeightedRandomSampler，使用 shuffle=True，避免与采样策略叠加）
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # 学习率调度器：OneCycleLR（按 batch 更新），需在 DataLoader 构建后初始化
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=10000.0
    )

    # 断点续训逻辑：先加载模型与优化器，再在此处恢复调度器状态
    start_epoch = 0
    best_acc = 0.0
    checkpoint = None
    if RESUME_CHECKPOINT and os.path.exists(LATEST_CHECKPOINT):
        checkpoint = load_checkpoint(
            LATEST_CHECKPOINT, 
            model, 
            optimizer, 
            None  # 调度器稍后恢复
        )
        if checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            early_stopper.best_acc = best_acc
            early_stopper.counter = checkpoint['early_stop_counter']
            visualizer.train_loss = checkpoint['train_loss_history']
            visualizer.val_loss = checkpoint['val_loss_history']
            visualizer.val_acc = checkpoint['val_acc_history']
            visualizer.val_acc_exits = checkpoint['val_acc_exits_history'] # 加载多个准确率历史
            # 加载动态推理历史（兼容旧检查点）
            visualizer.dynamic_acc = checkpoint.get('dynamic_acc_history', [])
            visualizer.exit_ratio_history = checkpoint.get('exit_ratio_history', [[] for _ in range(num_exits)])
            # 新增：动态FLOPs历史
            visualizer.dynamic_flops_g = checkpoint.get('dynamic_flops_history', [])
            current_alpha = checkpoint.get('current_alpha', ALPHA) # 加载当前的损失权重
            # 恢复调度器状态
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state'])
            except Exception:
                print("调度器状态与当前设置不匹配，已跳过恢复。")
            # 恢复EMA shadow
            ema_shadow = checkpoint.get('ema_shadow', None)
            if ema_shadow is not None:
                # 将shadow移到当前设备
                ema.shadow = {k: v.to(DEVICE) if hasattr(v, 'to') else v for k, v in ema_shadow.items()}
            else:
                ema = ModelEMA(model, decay=EMA_DECAY)
            # 恢复动态阈值
            thresholds_rt = checkpoint.get('thresholds', thresholds_rt)
            print(f"恢复训练：从epoch {start_epoch}开始，最佳精度 {best_acc:.4f}")

    # 训练循环
    for epoch in range(start_epoch, EPOCHS):
        # 两阶段切换
        if (not stage2_started) and (epoch + 1 >= STAGE1_EPOCHS):
            # 更聚焦最终出口
            current_alpha = [0.2] * (len(EXIT_BLOCKS)) + [1.0]
            kd_lambda = KD_LAMBDA_STAGE2 if KD_ENABLE else 0.0
            kd_ext_lambda = KD_EXT_LAMBDA_STAGE2 if (EXT_TEACHER_ENABLE and (ext_teacher is not None)) else 0.0
            stage2_started = True
        
        model.train()
        train_loss = 0.0
        
        # 训练阶段
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad(set_to_none=True)
            
            with (autocast() if DEVICE.type == 'cuda' else nullcontext()):
                # 关键改动：模型返回多个输出
                outputs_list = model(inputs) # [exit1_logits, exit2_logits, ..., final_logits]
                # 外部教师前向（可选）
                ext_teacher_logits = None
                if ext_teacher is not None:
                    with torch.no_grad():
                        ext_teacher_logits = ext_teacher(inputs)
                # 1) 交叉熵：各出口监督
                total_loss_ce = 0.0
                for i, (outputs, alpha) in enumerate(zip(outputs_list, current_alpha)):
                    loss_i = criterion(outputs, labels)
                    total_loss_ce += alpha * loss_i
                # 2) KD 蒸馏：用最终出口作为教师，指导早退头
                total_loss_kd = 0.0
                if KD_ENABLE and kd_lambda > 0.0:
                    teacher_logits = outputs_list[-1].detach()
                    for i in range(len(outputs_list) - 1):
                        total_loss_kd += kd_lambda * kd_kl_div(outputs_list[i], teacher_logits, KD_TEMPERATURE)
                # 3) 外部教师蒸馏：指导最终与早期出口
                if (ext_teacher_logits is not None) and (kd_ext_lambda > 0.0):
                    # 最终出口蒸馏
                    total_loss_kd += kd_ext_lambda * kd_kl_div(outputs_list[-1], ext_teacher_logits, KD_TEMPERATURE)
                    # 早期出口蒸馏（弱一些）
                    if KD_EXT_TO_EARLY:
                        for i in range(len(outputs_list) - 1):
                            total_loss_kd += (0.5 * kd_ext_lambda) * kd_kl_div(outputs_list[i], ext_teacher_logits, KD_TEMPERATURE)
                # 4) BN 稀疏正则（促进后续通道剪枝）
                if PRUNE_REG_ENABLE and PRUNE_REG_LAMBDA > 0 and bn_weights:
                    bn_l1 = 0.0
                    for w in bn_weights:
                        bn_l1 = bn_l1 + w.abs().sum()
                    total_loss = total_loss_ce + total_loss_kd + PRUNE_REG_LAMBDA * (bn_l1 / total_bn)
                else:
                    total_loss = total_loss_ce + total_loss_kd
            
            scaler.scale(total_loss).backward()
            # 梯度裁剪（需先反缩放）
            if DEVICE.type == 'cuda':
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            scaler.step(optimizer)
            scaler.update()
            # OneCycleLR 每个 batch 更新
            scheduler.step()
            # EMA更新（以优化器步进后参数为基准）
            ema.update(model)
            
            train_loss += total_loss.item() * inputs.size(0) # 使用总损失
        
        # 每个epoch衰减早期出口的损失权重（最终出口不衰减）
        for i in range(len(current_alpha) - 1):
            current_alpha[i] *= ALPHA_DECAY_FACTOR
        train_loss = train_loss / len(train_loader.dataset)

        # 验证阶段（使用EMA权重进行评估与选择最佳）
        ema.apply_shadow(model)
        model.eval()
        val_loss = 0.0
        correct_list = [0] * num_exits # 存储每个出口正确的数量
        total = 0
        # 动态推理统计
        dynamic_correct = 0
        exit_counts = [0] * num_exits
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs_list = model(inputs)
                # 验证阶段也只计算最终损失用于监控，但评估所有出口的准确率
                final_loss = criterion(outputs_list[-1], labels)
                val_loss += final_loss.item() * inputs.size(0)
                
                # 计算每个出口的准确率
                for i, outputs in enumerate(outputs_list):
                    _, preds = torch.max(outputs, 1)
                    correct_list[i] += torch.sum(preds == labels.data).item()
                
                # 动态推理：按置信度阈值决定是否提前退出
                batch_size = labels.size(0)
                total += batch_size
                for b in range(batch_size):
                    exited = False
                    for i in range(num_exits - 1):  # 遍历早退分支
                        probs = torch.softmax(outputs_list[i][b], dim=-1)
                        conf, pred = torch.max(probs, dim=-1)
                        if conf.item() >= thresholds_rt[i]:
                            if pred.item() == labels[b].item():
                                dynamic_correct += 1
                            exit_counts[i] += 1
                            exited = True
                            break
                    if not exited:
                        # 使用最终出口
                        probs_final = torch.softmax(outputs_list[-1][b], dim=-1)
                        pred_final = torch.argmax(probs_final).item()
                        if pred_final == labels[b].item():
                            dynamic_correct += 1
                        exit_counts[-1] += 1
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc_list = [c / total for c in correct_list] # 每个出口的准确率
        final_val_acc = val_acc_list[-1] # 最终出口的准确率
        dynamic_acc = dynamic_correct / total if total > 0 else 0.0
        exit_ratios = [c / total for c in exit_counts] if total > 0 else [0.0] * num_exits
        # 新增：计算期望动态FLOPs(G)
        if (len(exit_ratios) == num_exits) and (not any(np.isnan(flops_exits_g))) and (not np.isnan(flops_full_g)):
            costs = flops_exits_g + [flops_full_g]
            expected_dyn_flops_g = float(sum([exit_ratios[i] * costs[i] for i in range(num_exits)]))
        else:
            expected_dyn_flops_g = float('nan')
        
        # 阈值自动校准（在EMA评估权重下执行），尽量减少额外开销
        if AUTO_CALIBRATE_THRESHOLDS and (epoch + 1) >= CALIBRATION_START_EPOCH and ((epoch + 1 - CALIBRATION_START_EPOCH) % CALIBRATION_FREQ == 0):
            new_thr = calibrate_thresholds(model, val_loader, num_exits, thresholds_rt, DEVICE)
            if new_thr != thresholds_rt:
                print(f"阈值校准：{thresholds_rt} -> {new_thr}")
                thresholds_rt = new_thr
        # 还原回训练权重
        ema.restore(model)

        # 更新可视化（传入所有出口的准确率与动态推理统计）
        visualizer.update(epoch, train_loss, val_loss, final_val_acc, val_acc_list, dynamic_acc, exit_ratios, expected_dyn_flops_g)

        # 保存最佳模型（基于最终出口的准确率，保存EMA权重到 model_state）
        if final_val_acc > best_acc:
            best_acc = final_val_acc
            save_checkpoint({
                'epoch': epoch,
                'model_state': {k: v.cpu() for k, v in ema.shadow.items()},
                'model_state_raw': model.state_dict(),
                'class_names': class_names,
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'val_acc': final_val_acc, # 最终准确率
                'val_acc_exits': val_acc_list, # 所有出口准确率
                'val_acc_dynamic': dynamic_acc,
                'exit_ratios': exit_ratios,
                'expected_dyn_flops_g': expected_dyn_flops_g,
                'best_acc': best_acc,
                'early_stop_counter': early_stopper.counter,
                'train_loss_history': visualizer.train_loss,
                'val_loss_history': visualizer.val_loss,
                'val_acc_history': visualizer.val_acc,
                'val_acc_exits_history': visualizer.val_acc_exits, # 保存所有准确率历史
                'dynamic_acc_history': visualizer.dynamic_acc,
                'dynamic_flops_history': visualizer.dynamic_flops_g,
                'exit_ratio_history': visualizer.exit_ratio_history,
                'current_alpha': current_alpha, # 保存当前的损失权重
                'exit_blocks': EXIT_BLOCKS,
                'ema_shadow': {k: v.cpu() for k, v in ema.shadow.items()},
                'kd_hparams': {
                    'KD_ENABLE': KD_ENABLE,
                    'KD_TEMPERATURE': KD_TEMPERATURE,
                    'KD_LAMBDA_STAGE1': KD_LAMBDA_STAGE1,
                    'KD_LAMBDA_STAGE2': KD_LAMBDA_STAGE2,
                    'EXT_TEACHER_ENABLE': EXT_TEACHER_ENABLE,
                    'KD_EXT_LAMBDA_STAGE1': KD_EXT_LAMBDA_STAGE1,
                    'KD_EXT_LAMBDA_STAGE2': KD_EXT_LAMBDA_STAGE2,
                    'KD_EXT_TO_EARLY': KD_EXT_TO_EARLY,
                },
                'ema_decay': EMA_DECAY,
                'stage1_ratio': STAGE1_RATIO,
                'width_mult': WIDTH_MULT,
                'ext_teacher_used': EXT_TEACHER_ENABLE and (ext_teacher is not None),
                'ext_teacher_ckpt': EXT_TEACHER_CKPT,
                'thresholds': thresholds_rt,
                # 静态FLOPs信息
                'flops_full_g': flops_full_g,
                'flops_exits_g': flops_exits_g,
            }, BEST_CHECKPOINT)

        # 保存最新检查点（包含EMA shadow以支持断点续训）
        save_checkpoint({
            'epoch': epoch,
            'model_state': {k: v.cpu() for k, v in ema.shadow.items()},
            'model_state_raw': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'val_acc': final_val_acc,
            'val_acc_exits': val_acc_list,
            'val_acc_dynamic': dynamic_acc,
            'exit_ratios': exit_ratios,
            'expected_dyn_flops_g': expected_dyn_flops_g,
            'best_acc': best_acc,
            'early_stop_counter': early_stopper.counter,
            'train_loss_history': visualizer.train_loss,
            'val_loss_history': visualizer.val_loss,
            'val_acc_history': visualizer.val_acc,
            'val_acc_exits_history': visualizer.val_acc_exits,
            'dynamic_acc_history': visualizer.dynamic_acc,
            'dynamic_flops_history': visualizer.dynamic_flops_g,
            'exit_ratio_history': visualizer.exit_ratio_history,
            'current_alpha': current_alpha,
            'exit_blocks': EXIT_BLOCKS,
            'ema_shadow': {k: v.cpu() for k, v in ema.shadow.items()},
            'kd_hparams': {
                'KD_ENABLE': KD_ENABLE,
                'KD_TEMPERATURE': KD_TEMPERATURE,
                'KD_LAMBDA_STAGE1': KD_LAMBDA_STAGE1,
                'KD_LAMBDA_STAGE2': KD_LAMBDA_STAGE2,
            },
            'ema_decay': EMA_DECAY,
            'stage1_ratio': STAGE1_RATIO,
            'width_mult': WIDTH_MULT,
            'ext_teacher_used': EXT_TEACHER_ENABLE and (ext_teacher is not None),
            'ext_teacher_ckpt': EXT_TEACHER_CKPT,
            # 静态FLOPs信息
            'flops_full_g': flops_full_g,
            'flops_exits_g': flops_exits_g,
        }, LATEST_CHECKPOINT)

        # 早停检查（基于最终出口的准确率）
        if early_stopper.check(final_val_acc):
            print(f"\n早停触发于epoch {epoch+1}! 最佳精度: {early_stopper.best_acc:.4f}")
            break

        # 打印日志（显示所有出口的准确率与动态推理统计）
        current_lr = optimizer.param_groups[0]['lr']
        log_message = f"Epoch {epoch+1}/{EPOCHS} | LR: {current_lr:.6f} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
        for i, acc in enumerate(val_acc_list):
            exit_name = "Final" if i == len(val_acc_list)-1 else f"Exit{i+1}"
            log_message += f"{exit_name} Acc: {acc:.4f} | "
        # 新增：退出占比与期望动态FLOPs
        log_message += f"Dynamic Acc: {dynamic_acc:.4f} | Exit Ratios: {[round(r,3) for r in exit_ratios]} | Dyn FLOPs(G): {expected_dyn_flops_g:.4f}"
        print(log_message)

if __name__ == "__main__":
    main()