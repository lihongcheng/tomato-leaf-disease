import cv2
import albumentations as A
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import random
from collections import OrderedDict
import matplotlib.font_manager as fm

# --- 中文字体设置 ---
# 为了在图中正确显示中文，请确保您的系统上存在一个支持中文的字体文件。
# SimHei.ttf 是一个常用的黑体字，您也可以替换为其他字体（如 ariy.ttf 等）。
# 如果脚本在您的服务器上找不到字体，请将字体文件上传到服务器，并在此处提供正确路径。
try:
    font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'  # 常见 Linux 路径
    cn_font = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = cn_font.get_name()
    plt.rcParams['axes.unicode_minus'] = False # 正确显示负号
    print(f"成功加载中文字体: {cn_font.get_name()}")
except Exception as e:
    print(f"加载中文字体失败，图像中的中文可能无法正常显示: {e}")
    print("请确认您的系统上已安装中文字体，或修改脚本中的 font_path。")
    cn_font = None


# --- 配置 (请在您的训练服务器上确认这些路径) ---
DATASET_ROOT = "/toyoudatasetrpath/" #数据集根目录
CSV_PATH = os.path.join(DATASET_ROOT, 'metadata/master.csv')
OUTPUT_FILENAME = 'Fig2-2_data_augmentation_examples_corrected.png'
TARGET_SIZE = (224, 224)
SEED = 42

# 设置随机种子以保证结果可复现
random.seed(SEED)
np.random.seed(SEED)
# Albumentations 也需要设置随机种子
A.Compose([A.HorizontalFlip(p=0.5)], p=1.0)


# --- 辅助函数 ---
def load_image(image_path):
    """使用OpenCV加载图像并转换为RGB格式。"""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法加载图像: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def get_visualization_transforms():
    """
    获取用于可视化的数据增强变换字典。
    - 每个变换都强制应用 (p=1.0) 以清晰展示其效果。
    - 所有变换都包含最终的尺寸调整，确保输出图像大小一致。
    - 包含一个与 data_process.py 逻辑一致的组合变换。
    - 兼容新旧版本的 albumentations 库，以解决 pydantic 验证错误。
    """
    # resize_op = A.Resize(height=TARGET_SIZE[0], width=TARGET_SIZE[1]) # 不再需要全局resize

    # --- 创建兼容新旧库版本的 RandomResizedCrop ---
    # 这是为了解决在较新版本的 albumentations 中，RandomResizedCrop
    # 需要 `size` 参数而不是 `height` 和 `width`，从而引发的 pydantic ValidationError。
    try:
        # 新版本 albumentations (>1.4.0)
        random_crop_op = A.RandomResizedCrop(
            size=TARGET_SIZE, scale=(0.6, 1.0), ratio=(0.75, 1.33), p=1.0
        )
        combined_crop_op = A.RandomResizedCrop(
            size=TARGET_SIZE, scale=(0.6, 1.0), ratio=(0.75, 1.33), p=0.9
        )
    except TypeError:
        # 旧版本 albumentations
        random_crop_op = A.RandomResizedCrop(
            height=TARGET_SIZE[0], width=TARGET_SIZE[1], scale=(0.6, 1.0), ratio=(0.75, 1.33), p=1.0
        )
        combined_crop_op = A.RandomResizedCrop(
            height=TARGET_SIZE[0], width=TARGET_SIZE[1], scale=(0.6, 1.0), ratio=(0.75, 1.33), p=0.9
        )
    # ---

    # 1. 单独的、用于清晰可视化的变换（保持原图尺寸，统一缩放在可视化阶段完成）
    transforms = OrderedDict()
    transforms["1. 随机剪切与缩放"] = A.Compose([
        random_crop_op
    ])
    transforms["2. 增加水平翻转"] = A.Compose([
        A.HorizontalFlip(p=1.0),
    ])
    transforms["3. 增加颜色抖动"] = A.Compose([
        # 参数稍微夸张以便于观察
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2, p=1.0),
    ])
    
    # 2. 模拟真实训练的组合变换 (源自 data_process.py)
    # 不包含归一化（Normalize）和ToTensorV2，以便于可视化
    combined_transform = A.Compose([
        combined_crop_op,
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, p=0.8),
    ])
     
    # 将组合变换添加到字典中，并确保顺序与示例图一致
    transforms["4. 完整组合增强（示例1）"] = combined_transform
    transforms["5. 完整组合增强（示例2）"] = combined_transform
    
    return transforms

def visualize_augmentations(image, transforms_dict):
    """应用变换并使用Matplotlib绘制结果。"""
    # +1 是为了包含原始图像
    num_images = len(transforms_dict) + 1

    # 改为更紧凑的网格布局，避免图像过宽
    ncols = 3  # 每行最多显示3张图
    nrows = (num_images + ncols - 1) // ncols  # 自动计算所需行数

    # 根据新的布局调整画布大小 (每张子图约2.4x2.6英寸，更紧凑)
    fig = plt.figure(figsize=(2.4 * ncols, 2.6 * nrows))

    # 添加一个居中的大标题
    fig.suptitle(
        '训练时数据增强策略效果对比（修正版）',
        fontsize=14,          # 更小的主标题字号
        fontweight='bold',    # 加粗
        y=0.985,              # 更靠近顶部
        fontproperties=cn_font
    )

    # 1. 显示原始图像 (统一调整大小以便公平比较)
    plt.subplot(nrows, ncols, 1)
    # 使用 INTER_AREA 插值以在缩小时获得更好的效果
    resized_image = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    plt.imshow(resized_image)
    plt.title('原始图像 (Resized)', fontproperties=cn_font, fontsize=10, pad=6)
    plt.axis('off')

    # 2. 显示经过各种增强的图像
    for i, (name, transform) in enumerate(transforms_dict.items()):
        # 重置随机种子，确保"组合示例"每次都不同
        np.random.seed(SEED + i)

        # 应用变换
        augmented_image = transform(image=image)['image']

        # 在显示前，统一将所有图像（包括未被albumentations调整大小的）调整到目标尺寸
        augmented_image_resized = cv2.resize(augmented_image, TARGET_SIZE, interpolation=cv2.INTER_AREA)

        ax = plt.subplot(nrows, ncols, i + 2)
        ax.imshow(augmented_image_resized)
        ax.set_title(name, fontproperties=cn_font, fontsize=9, pad=6)
        ax.axis('off')

    # 使用更细致的间距控制，避免文字覆盖图像
    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.06, top=0.86, wspace=0.25, hspace=0.55)
    plt.savefig(OUTPUT_FILENAME, dpi=150)
    print(f"对比图已成功保存到: {OUTPUT_FILENAME}")
    # 在非交互式环境中，注释掉 plt.show()
    # plt.show()

# --- 主程序 ---
if __name__ == "__main__":
    try:
        # 1. 加载元数据
        df = pd.read_csv(CSV_PATH)
        
        # 2. 筛选一张健康的叶片图像用于演示
        # 使用 'Healthy' 标签和 'orig_path' 列，与 data_process.py 保持一致
        healthy_leaves = df[df['class'] == 'Healthy']
        if healthy_leaves.empty:
            raise ValueError(f"在 {CSV_PATH} 中未找到 'class' 列为 'Healthy' 的样本。")
        
        # 随机选择一个样本
        sample_row = healthy_leaves.sample(1, random_state=SEED).iloc[0]
        image_relative_path = sample_row['orig_path']
        image_path = os.path.join(DATASET_ROOT, image_relative_path)

        # 3. 加载图像
        original_image = load_image(image_path)

        # 4. 获取变换并执行可视化
        transforms_to_show = get_visualization_transforms()
        visualize_augmentations(original_image, transforms_to_show)

    except FileNotFoundError as e:
        print(f"错误: 文件或路径未找到。请在您的服务器上确认以下路径是否正确: ")
        print(f"  - DATASET_ROOT: {DATASET_ROOT}")
        print(f"  - CSV_PATH: {CSV_PATH}")
        print(f"  - 图像路径: {e.filename}")
    except ValueError as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")