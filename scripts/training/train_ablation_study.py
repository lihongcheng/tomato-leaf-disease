import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, StepLR
from torch.cuda.amp import GradScaler, autocast
from data_process import DataProcessor, PlantDataset
import torchvision.models as models
import os
import numpy as np
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Ablation study for training strategies.")
    parser.add_argument('--experiment_name', type=str, required=True, help='Name for the experiment, used for output directory.')
    parser.add_argument('--loss_function', type=str, default='labelsmooth', choices=['labelsmooth', 'crossentropy'], help='Loss function to use.')
    parser.add_argument('--sampler', type=str, default='weighted', choices=['weighted', 'standard'], help='Data sampler to use.')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step'], help='Learning rate scheduler to use.')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd'], help='Optimizer to use.')
    return parser.parse_args()

# 配置参数
plt.switch_backend('Agg')

class TrainingVisualizer:
    """训练过程可视化器"""
    def __init__(self, results_dir):
        self.train_loss = []
        self.val_loss = []
        self.val_acc = []
        self.fig, self.ax = plt.subplots(2, 1, figsize=(10, 10))
        self.fig_path = os.path.join(results_dir, "training_curve.png")
        self.log_path = os.path.join(results_dir, "training_log.csv")
        
    def update(self, epoch, t_loss, v_loss, v_acc):
        self.train_loss.append(t_loss)
        self.val_loss.append(v_loss)
        self.val_acc.append(v_acc)
        
        self.ax[0].clear()
        self.ax[0].plot(range(1, epoch+2), self.train_loss, label='Train Loss')
        self.ax[0].plot(range(1, epoch+2), self.val_loss, label='Val Loss')
        self.ax[0].set_title('Loss Curve')
        self.ax[0].legend()
        
        self.ax[1].clear()
        self.ax[1].plot(range(1, epoch+2), self.val_acc, 'g-', label='Val Acc')
        self.ax[1].set_title('Accuracy Curve')
        self.ax[1].set_ylim(0.5, 1.0)
        self.ax[1].legend()
        
        self.fig.tight_layout()
        self.fig.savefig(self.fig_path)
        np.savetxt(self.log_path, 
                  np.column_stack([self.train_loss, self.val_loss, self.val_acc]),
                  delimiter=',',
                  header='train_loss,val_loss,val_acc',
                  comments='')

class EarlyStopper:
    """早停控制器"""
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_acc = 0.0
        self.counter = 0
        self.early_stop = False
        
    def check(self, val_acc):
        if val_acc > self.best_acc + self.min_delta:
            self.best_acc = val_acc
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

class LabelSmoothCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
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

def save_checkpoint(state, filename):
    torch.save(state, filename)
    print(f"=> 保存检查点到 {filename}")

def main():
    args = get_args()

    # 根据实验名称创建结果目录
    RESULTS_DIR = f"./results/{args.experiment_name}"
    CHECKPOINT_DIR = f"./checkpoints/{args.experiment_name}"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    BEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_checkpoint.pth")

    # 初始化组件
    processor = DataProcessor()
    visualizer = TrainingVisualizer(RESULTS_DIR)
    early_stopper = EarlyStopper()
    
    # 数据加载
    master_df = processor.generate_metadata()
    train_set = PlantDataset(master_df[master_df['split'] == 'train'], mode='train')
    val_set = PlantDataset(master_df[master_df['split'] == 'val'], mode='val')
    
    # 模型初始化
    model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
    model.classifier[3] = nn.Linear(1024, len(train_set.class_map))
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(DEVICE)

    # 根据参数选择损失函数
    if args.loss_function == 'labelsmooth':
        criterion = LabelSmoothCrossEntropy(smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss()

    # 根据参数选择优化器
    if args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    # 根据参数选择学习率调度器
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    else:
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
        
    scaler = GradScaler()

    # 根据参数选择数据采样器
    if args.sampler == 'weighted':
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=1.0 / master_df[master_df['split'] == 'train']['class'].value_counts(
                normalize=True)[master_df[master_df['split'] == 'train']['class']].values,
            num_samples=len(train_set),
            replacement=True)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=128, sampler=sampler, num_workers=14, pin_memory=True, persistent_workers=True
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=128, shuffle=True, num_workers=14, pin_memory=True, persistent_workers=True
        )
    
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=128, shuffle=False, num_workers=14, pin_memory=True
    )

    best_acc = 0.0
    # 训练循环
    for epoch in range(100):
        model.train()
        train_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * inputs.size(0)
        
        scheduler.step()
        train_loss /= len(train_loader.sampler) if args.sampler == 'weighted' else len(train_loader.dataset)

        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data)
        
        val_loss /= len(val_loader.dataset)
        val_acc = correct.double() / len(val_loader.dataset)

        visualizer.update(epoch, train_loss, val_loss, val_acc.item())

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint({'epoch': epoch, 'model_state': model.state_dict()}, BEST_CHECKPOINT)

        if early_stopper.check(val_acc):
            print(f"\n早停触发于epoch {epoch+1}! 最佳精度: {early_stopper.best_acc:.4f}")
            break

        print(f"Epoch {epoch+1}/100 | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Best Acc: {best_acc:.4f}")

if __name__ == "__main__":
    main()