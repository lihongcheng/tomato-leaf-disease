import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
from data_process import DataProcessor, PlantDataset
import torchvision.models as models
import os
import numpy as np
from tqdm import tqdm
from attention_module import CoordAtt
from torchvision.models import mobilenet_v3_small

# 1. Model Definition with Attention
class MobileNetV3WithAttention(nn.Module):
    def __init__(self, num_classes=10, attention_channels=24):
        super(MobileNetV3WithAttention, self).__init__()
        # Load pre-trained MobileNetV3 Small
        self.mobilenet = mobilenet_v3_small(pretrained=True)

        # Extract features part
        self.features = self.mobilenet.features

        # Add our Coordinate Attention module
        # We insert it after a specific layer, e.g., features[3]
        # The channel number (24) needs to match the output of features[3]
        self.attention = CoordAtt(inp=attention_channels, oup=attention_channels)

        # Re-construct the rest of the model
        self.avgpool = self.mobilenet.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        # Pass through the first few layers
        x = self.features[:4](x)
        # Apply attention
        x = self.attention(x)
        # Pass through the rest of the features
        x = self.features[4:](x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# 配置参数
plt.switch_backend('Agg')
RESULTS_DIR = "./results/MobileNetV3_Attention"
FIG_PATH = os.path.join(RESULTS_DIR, "training_curve.png")
LOG_PATH = os.path.join(RESULTS_DIR, "training_log.csv")
CHECKPOINT_DIR = "./checkpoints/MobileNetV3_Attention"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 检查点配置
RESUME_CHECKPOINT = True  # 是否启用断点续训
LATEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")
BEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_checkpoint.pth")

# 早停配置
EARLY_STOP_PATIENCE = 10
MIN_DELTA = 0.001

# 硬件配置
BATCH_SIZE = 128
NUM_WORKERS = 14
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TrainingVisualizer:
    """训练过程可视化器"""
    def __init__(self):
        self.train_loss = []
        self.val_loss = []
        self.val_acc = []
        self.fig, self.ax = plt.subplots(2, 1, figsize=(10, 10))
        
    def update(self, epoch, t_loss, v_loss, v_acc):
        # 记录数据
        self.train_loss.append(t_loss)
        self.val_loss.append(v_loss)
        self.val_acc.append(v_acc)
        
        # 绘制损失曲线
        self.ax[0].clear()
        self.ax[0].plot(range(1, epoch+2), self.train_loss, label='Train Loss')
        self.ax[0].plot(range(1, epoch+2), self.val_loss, label='Val Loss')
        self.ax[0].set_title('Loss Curve')
        self.ax[0].legend()
        
        # 绘制准确率曲线
        self.ax[1].clear()
        self.ax[1].plot(range(1, epoch+2), self.val_acc, 'g-', label='Val Acc')
        self.ax[1].set_title('Accuracy Curve')
        self.ax[1].set_ylim(0.5, 1.0)
        self.ax[1].legend()
        
        # 保存图片和CSV日志
        self.fig.tight_layout()
        self.fig.savefig(FIG_PATH)
        np.savetxt(LOG_PATH, 
                  np.column_stack([self.train_loss, self.val_loss, self.val_acc]),
                  delimiter=',',
                  header='train_loss,val_loss,val_acc',
                  comments='')

class EarlyStopper:
    """早停控制器"""
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
    """保存训练状态检查点"""
    torch.save(state, filename)
    print(f"=> 保存检查点到 {filename}")

def load_checkpoint(filename, model, optimizer=None, scheduler=None):
    """加载训练状态检查点"""
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state'])
        print(f"=> 从 {filename} 加载检查点 (epoch {checkpoint['epoch']})")
        return checkpoint
    else:
        print(f"=> 未找到检查点 {filename}")
        return None

def main():
    # 初始化组件
    processor = DataProcessor()
    visualizer = TrainingVisualizer()
    early_stopper = EarlyStopper()
    
    # 数据加载
    master_df = processor.generate_metadata()
    train_set = PlantDataset(master_df[master_df['split'] == 'train'], mode='train')
    val_set = PlantDataset(master_df[master_df['split'] == 'val'], mode='val')
    
    # 模型初始化
    model = MobileNetV3WithAttention(num_classes=len(train_set.class_map))
    model = model.to(DEVICE)

    # 优化器初始化
    criterion = LabelSmoothCrossEntropy(smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    scaler = GradScaler()

    # 断点续训逻辑
    start_epoch = 0
    best_acc = 0.0
    if RESUME_CHECKPOINT and os.path.exists(LATEST_CHECKPOINT):
        checkpoint = load_checkpoint(
            LATEST_CHECKPOINT, 
            model, 
            optimizer, 
            scheduler
        )
        if checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            best_acc = checkpoint['best_acc']
            early_stopper.best_acc = best_acc
            early_stopper.counter = checkpoint['early_stop_counter']
            visualizer.train_loss = checkpoint['train_loss_history']
            visualizer.val_loss = checkpoint['val_loss_history']
            visualizer.val_acc = checkpoint['val_acc_history']
            print(f"恢复训练：从epoch {start_epoch}开始，最佳精度 {best_acc:.4f}")

    # 数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        sampler=torch.utils.data.WeightedRandomSampler(
            weights=1.0 / master_df[master_df['split'] == 'train']['class'].value_counts(
                normalize=True)[master_df[master_df['split'] == 'train']['class']].values,
            num_samples=len(train_set),
            replacement=True),
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

    # 训练循环
    for epoch in range(start_epoch, 100):
        model.train()
        train_loss = 0.0
        
        # 训练阶段
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * inputs.size(0)
        
        scheduler.step()
        train_loss = train_loss / len(train_loader.dataset)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct.double() / len(val_loader.dataset)

        # 更新可视化
        visualizer.update(epoch, train_loss, val_loss, val_acc.item())

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'class_names': sorted(train_set.class_map.keys()),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'val_acc': val_acc,
                'best_acc': best_acc,
                'early_stop_counter': early_stopper.counter,
                'train_loss_history': visualizer.train_loss,
                'val_loss_history': visualizer.val_loss,
                'val_acc_history': visualizer.val_acc
            }, BEST_CHECKPOINT)

        # 保存最新检查点
        save_checkpoint({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_acc': best_acc,
            'early_stop_counter': early_stopper.counter,
            'train_loss_history': visualizer.train_loss,
            'val_loss_history': visualizer.val_loss,
            'val_acc_history': visualizer.val_acc
        }, LATEST_CHECKPOINT)

        # 早停检查
        if early_stopper.check(val_acc):
            print(f"\n早停触发于epoch {epoch+1}! 最佳精度: {early_stopper.best_acc:.4f}")
            break

        # 打印日志
        print(f"Epoch {epoch+1}/100 | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Best Acc: {best_acc:.4f}")

    print("训练完成！")

if __name__ == '__main__':
    main()
