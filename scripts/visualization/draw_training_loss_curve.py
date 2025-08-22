#!/usr/bin/env python3
"""
基于训练日志绘制五个模型的训练损失收敛曲线
生成Fig5-4_training_loss_convergence.svg
"""

import re
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体和支持负号显示
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def parse_efficientnet_log(file_path):
    """解析EfficientNet训练日志"""
    epochs = []
    train_losses = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 匹配 Epoch X/200 - Train Loss: Y.ZZZZ
    pattern = r'Epoch\s+(\d+)/(?:\d+)\s*\|\s*Train Loss:\s*([\d\.eE+-]+)'
    matches = re.findall(pattern, content)
    
    for epoch, loss in matches:
        epochs.append(int(epoch))
        train_losses.append(float(loss))
    
    return epochs, train_losses

def parse_mobilenet_log(file_path):
    """解析MobileNet训练日志"""
    epochs = []
    train_losses = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 匹配 Epoch X/200 - Train Loss: Y.ZZZZ
    pattern = r'Epoch\s+(\d+)/(?:\d+)\s*\|\s*Train Loss:\s*([\d\.eE+-]+)'
    matches = re.findall(pattern, content)
    
    for epoch, loss in matches:
        epochs.append(int(epoch))
        train_losses.append(float(loss))
    
    return epochs, train_losses

def parse_shufflenet_log(file_path):
    """解析ShuffleNet训练日志"""
    epochs = []
    train_losses = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 匹配 Epoch X/200 - Train Loss: Y.ZZZZ
    pattern = r'Epoch\s+(\d+)/(?:\d+)\s*\|\s*Train Loss:\s*([\d\.eE+-]+)'
    matches = re.findall(pattern, content)
    
    for epoch, loss in matches:
        epochs.append(int(epoch))
        train_losses.append(float(loss))
    
    return epochs, train_losses

def parse_mobilenetv3_log(file_path):
    """解析MobileNetV3训练日志"""
    epochs = []
    train_losses = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 匹配 Epoch X/200 - Train Loss: Y.ZZZZ
    pattern = r'Epoch\s+(\d+)/(?:\d+)\s*\|\s*Train Loss:\s*([\d\.eE+-]+)'
    matches = re.findall(pattern, content)
    
    for epoch, loss in matches:
        epochs.append(int(epoch))
        train_losses.append(float(loss))
    
    return epochs, train_losses

def parse_mobilenetv3_attention_log(file_path):
    """解析MobileNetV3 with Attention训练日志"""
    epochs = []
    train_losses = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 匹配 Epoch X/200 - Train Loss: Y.ZZZZ
    pattern = r'Epoch\s+(\d+)/(?:\d+)\s*\|\s*Train Loss:\s*([\d\.eE+-]+)'
    matches = re.findall(pattern, content)
    
    for epoch, loss in matches:
        epochs.append(int(epoch))
        train_losses.append(float(loss))
    
    return epochs, train_losses

def plot_training_loss_curves():
    """绘制训练损失收敛曲线"""
    
    # 日志文件路径
    log_files = {
        'EfficientNet-B0': 'train_efficientnetb0.output',
        'MobileNetV1': 'train_mobilenetv1.log',
        'MobileNetV3': 'train_mobilenetv3.output',
        'MobileNetV3+Attention': 'train_mobilenetv3_with_attention.log',
        'ShuffleNetV2': 'train_ShuffleNetV2.output'
    }
    
    # 解析器映射
    parsers = {
        'EfficientNet-B0': parse_efficientnet_log,
        'MobileNetV1': parse_mobilenet_log,
        'MobileNetV3': parse_mobilenetv3_log,
        'MobileNetV3+Attention': parse_mobilenetv3_attention_log,
        'ShuffleNetV2': parse_shufflenet_log
    }
    
    # 颜色和线型配置
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    linestyles = ['-', '--', '-.', ':', '-']
    markers = ['o', 's', '^', 'D', 'v']
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 解析并绘制每个模型的损失曲线
    for i, (model_name, log_file) in enumerate(log_files.items()):
        try:
            epochs, train_losses = parsers[model_name](log_file)
            
            if epochs and train_losses:
                # 每隔5个epoch标记一个点以减少密度
                step = max(1, len(epochs) // 40)
                epochs_marked = epochs[::step]
                losses_marked = train_losses[::step]
                
                # 绘制完整曲线（细线）
                ax.plot(epochs, train_losses, 
                       color=colors[i], 
                       linestyle=linestyles[i],
                       linewidth=1.5,
                       alpha=0.8,
                       label=model_name)
                
                # 添加标记点
                ax.plot(epochs_marked, losses_marked,
                       color=colors[i],
                       marker=markers[i],
                       markersize=4,
                       markevery=1,
                       alpha=0.9)
                
                print(f"✅ {model_name}: {len(epochs)} epochs, loss range: {min(train_losses):.4f} - {max(train_losses):.4f}")
            else:
                print(f"❌ {model_name}: No data found in {log_file}")
                
        except FileNotFoundError:
            print(f"❌ {model_name}: Log file {log_file} not found")
        except Exception as e:
            print(f"❌ {model_name}: Error parsing {log_file}: {e}")
    
    # 设置图表样式
    ax.set_xlabel('训练轮数 (Epoch)', fontsize=14, fontweight='bold')
    ax.set_ylabel('训练损失 (Training Loss)', fontsize=14, fontweight='bold')
    ax.set_title('五种模型训练损失收敛曲线对比', fontsize=16, fontweight='bold', pad=20)
    
    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 设置图例
    ax.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    # 设置Y轴范围（根据数据动态调整）
    ax.set_ylim(bottom=0)
    
    # 设置X轴范围
    ax.set_xlim(left=0)
    
    # 优化布局
    plt.tight_layout()
    
    # 保存为SVG格式
    output_path = 'Fig5-4_training_loss_convergence.svg'
    plt.savefig(output_path, format='svg', dpi=300, bbox_inches='tight')
    print(f"✅ 训练损失曲线已保存到: {output_path}")
    
    # 显示图表信息
    plt.show()

if __name__ == "__main__":
    plot_training_loss_curves()