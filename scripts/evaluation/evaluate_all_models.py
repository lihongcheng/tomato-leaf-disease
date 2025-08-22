import os
import time
import torch
import torch.nn as nn
import torchvision.models as tv_models
from train_mobilenetv3_with_attention import MobileNetV3WithAttention  # 导入新模型
import timm
import numpy as np
from ptflops import get_model_complexity_info
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from data_process import DataProcessor, PlantDataset
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------- 配置 -----------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
NUM_WORKERS = 8
IMAGE_SIZE = (224, 224)
MASTER_CSV = "/toyoudatasetrpath/metadata/master.csv" #元数据目录
CHECKPOINTS = {
    "MobileNetV3_small": "./checkpoints/MobileNetV3/best_checkpoint.pth",
    "EfficientNet_B0": "./checkpoints/efficientnetb0/best_checkpoint.pth",
    "MobileNetV1": "./checkpoints/MobileNetV1/best_checkpoint.pth",
    "ShuffleNetV2_x1_0": "./checkpoints/shufflenetV2/best_checkpoint.pth",
    "MobileNetV3_Attention": "./checkpoints/MobileNetV3_Attention/best_checkpoint.pth"  # 新增带注意力机制的模型
}

# 准备测试集
processor = DataProcessor()
master_df = processor.generate_metadata()
test_df = master_df[master_df['split'] == 'test']
test_dataset = PlantDataset(test_df, mode='val')
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)

# 通用加载与评估函数

def plot_confusion_matrix(y_true, y_pred, class_names, filename='confusion_matrix.svg'):
    """Plots and saves a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    print("\n--- Confusion Matrix (MobileNetV3-Small) ---")
    print("Class Names:", class_names)
    print("Matrix:\n", cm)
    print("--------------------------------------------\n")
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': '样本数量 (Count)'})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(filename, format='svg', bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {filename}")


def load_model(name, num_classes):
    if name == "MobileNetV3_small":
        m = tv_models.mobilenet_v3_small(weights=None)
        in_f = m.classifier[3].in_features
        m.classifier[3] = nn.Linear(in_f, num_classes)
    elif name == "EfficientNet_B0":
        m = tv_models.efficientnet_b0(weights=None)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, num_classes)
    elif name == "MobileNetV1":
        m = timm.create_model('mobilenetv1_100', pretrained=False)
        in_f = m.get_classifier().in_features
        m.classifier = nn.Linear(in_f, num_classes)
    elif name == "ShuffleNetV2_x1_0":  # 新增ShuffleNet加载逻辑
        m = tv_models.shufflenet_v2_x1_0(weights=None)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
    elif name == "MobileNetV3_Attention": # 新增自定义模型加载逻辑
        m = MobileNetV3WithAttention(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")
    return m.to(DEVICE)


def evaluate(model, checkpoint_path):
    # 加载权重与训练元信息
    ck = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ck.get('model_state', ck.get('model', {})))
    epoch = ck.get('epoch', None)
    best_acc = ck.get('best_acc', ck.get('best', None))

    # 参数量
    params = sum(p.numel() for p in model.parameters())

    # FLOPs
    macs, params_flops = get_model_complexity_info(
        model, (3, IMAGE_SIZE[0], IMAGE_SIZE[1]), as_strings=False, print_per_layer_stat=False)

    # 推理速度测量（单张）
    model.eval()
    dummy = torch.randn(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1], device=DEVICE)
    # 预热
    for _ in range(10): _ = model(dummy)
    # 计时
    iters = 50
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(iters): _ = model(dummy)
    torch.cuda.synchronize()
    latency = (time.time() - start) / iters  # seconds per inference

    # 测试集准确率
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            out = model(x)
            preds = out.argmax(1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(y.numpy())
    test_acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    return {
        'epochs_trained': epoch,
        'best_val_acc': best_acc,
        'test_acc': test_acc,
        'f1_score': f1,
        'param_count': params,
        'flops': macs * 2,  # 将 MACs 转换为 FLOPs
        'latency_s': latency,
        'throughput_img_per_s': 1.0 / latency,
        'model_size_MB': os.path.getsize(checkpoint_path) / (1024 * 1024),
        'y_true': y_true,
        'y_pred': y_pred
    }


def main():
    results = {}
    num_classes = len(test_dataset.class_map)
    class_names = list(test_dataset.class_map.keys())
    for name, ckpt in CHECKPOINTS.items():
        print(f"Evaluating {name}...")
        model = load_model(name, num_classes)
        stats = evaluate(model, ckpt)
        results[name] = stats

        # For MobileNetV3_small, generate and save the confusion matrix
        if name == "MobileNetV3_small":
            plot_confusion_matrix(
                stats['y_true'],
                stats['y_pred'],
                class_names,
                filename='Fig5-3_confusion_matrix_mobilenetv3_small.svg'
            )

    # 打印对比表格
    from tabulate import tabulate
    headers = ["Model", "Epochs", "Val Acc", "Test Acc", "F1-Score", "#Params", "FLOPs", "Latency(s)", "Throughput(img/s)", "Size(MB)"]
    table = []
    for name, s in results.items():
        table.append([
            name, s['epochs_trained'], f"{s['best_val_acc']:.4f}", f"{s['test_acc']:.4f}", f"{s['f1_score']:.4f}",
            s['param_count'], f"{s['flops']:.2e}", f"{s['latency_s']:.4f}", f"{s['throughput_img_per_s']:.1f}",
            f"{s['model_size_MB']:.2f}"
        ])
    print(tabulate(table, headers=headers, tablefmt='github'))

if __name__ == "__main__":
    main()
