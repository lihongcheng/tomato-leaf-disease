import os
import uuid
import json
import numpy as np
from PIL import Image
import pandas as pd
import albumentations as A
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import ClusterCentroids
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from collections import defaultdict
from albumentations.pytorch import ToTensorV2
import torch

# 全局配置
class Config:
    DATA_ROOT = "/root/autodl-tmp/tomato/dataset"
    AUG_DIR = os.path.join(DATA_ROOT, "augmented")
    META_DIR = os.path.join(DATA_ROOT, "metadata")
    SPLIT_RATIO = (0.7, 0.15, 0.15)
    SEED = 42
    TARGET_SIZE = (224, 224)
    # 更新后的采样策略
    SAMPLE_STRATEGY = {
        'Yellow_Leaf_Curl_Virus': 8000,  # 欠采样
        'Powdery_Mildew': 5000,          # 过采样
        'Target_Spot': 4500,             # 过采样
        'Spider_mites_Two_spotted_spider_mite': 5000, # 过采样
        '_default': 'original'
    }
    TRAIN_SCALE = (0.6, 1.0)   # 训练时裁剪比例
    AUG_SCALE = (0.8, 1.0)    # 增强时裁剪比例

# 初始化目录
os.makedirs(Config.AUG_DIR, exist_ok=True)
os.makedirs(Config.META_DIR, exist_ok=True)

class DataProcessor:
    def __init__(self):
        self.master_meta = os.path.join(Config.META_DIR, "master.csv")
        self.class_encoder = {}
        
    def generate_metadata(self):
        if os.path.exists(self.master_meta):
            return pd.read_csv(self.master_meta)
            
        # 收集原始数据
        meta_records = []
        original_dir = os.path.join(Config.DATA_ROOT, "original")
        for cls in os.listdir(original_dir):
            cls_dir = os.path.join(original_dir, cls)
            if os.path.isdir(cls_dir):
                for fname in os.listdir(cls_dir):
                    meta_records.append({
                        'orig_path': os.path.join("original", cls, fname),
                        'aug_path': None,
                        'class': cls,
                        'split': 'unassigned',
                        'is_augmented': False
                    })
        
        # 应用采样策略
        df = self._apply_sampling(pd.DataFrame(meta_records))
        df = self._split_dataset(df)
        df.to_csv(self.master_meta, index=False)
        return df
    
    def _apply_sampling(self, df):
        processed = []
        
        for cls, group in df.groupby('class'):
            target = Config.SAMPLE_STRATEGY.get(cls, Config.SAMPLE_STRATEGY['_default'])
            
            if isinstance(target, int):
                current_count = len(group)
                if current_count < target:
                    augmented = self._oversample(cls, group, target)
                    processed.append(augmented)
                elif current_count > target:
                    sampled = self._undersample(group, target)
                    processed.append(sampled)
                else:
                    processed.append(group)
            else:
                processed.append(group)
        
        # 修正合并方式，重置全局索引
        return pd.concat(processed, axis=0, ignore_index=True)
    
    def _oversample(self, cls, group, target):
        need = target - len(group)
        indices = np.random.choice(group.index, size=need, replace=True)
        
        new_records = []
        for idx in indices:
            orig_path = group.loc[idx, 'orig_path']
            aug_path = os.path.join("augmented", cls, f"aug_{uuid.uuid4().hex}.jpg")
            
            self._generate_augmented_image(
                os.path.join(Config.DATA_ROOT, orig_path),
                os.path.join(Config.DATA_ROOT, aug_path)
            )
            
            new_records.append({
                'orig_path': orig_path,
                'aug_path': aug_path,
                'class': cls,
                'split': 'unassigned',
                'is_augmented': True
            })
        
        return pd.concat([group, pd.DataFrame(new_records)])
    
    # 修改 _undersample 方法
    def _undersample(self, group, target):
        """稳健的欠采样方法"""
        group = group.reset_index(drop=True)  # 确保索引连续
        if len(group) <= target:
            return group

        # 直接使用随机采样，因为ClusterCentroids不适用于单类别降采样
        sampled = group.sample(n=target, random_state=Config.SEED)

        # 最终验证
        self._validate_sampling(group, sampled)
        return sampled.reset_index(drop=True)
    
    def _validate_sampling(self, group, sampled):
        """增强版采样验证"""
        if len(sampled) == 0:
            raise ValueError(f"{group['class'].iloc[0]}采样后样本为空")
        if sampled['class'].nunique() != 1:
            raise ValueError(f"{group['class'].iloc[0]}采样导致类别混合")
        print(f"采样成功: {group['class'].iloc[0]} {len(group)} -> {len(sampled)}")
    
    def _generate_augmented_image(self, src_path, dst_path):
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        img = Image.open(src_path).convert('RGB')

        # 应用增强
        aug = self._get_augmentor(os.path.basename(os.path.dirname(dst_path)))
        augmented = aug(image=np.array(img))['image']

        # 转换为numpy数组并调整通道顺序（修复变量名）
        augmented_np = augmented.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC

        # 反归一化并转换到0-255范围
        augmented_np = (augmented_np * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
        augmented_np = augmented_np.astype(np.uint8)

        # 修改保存格式为JPEG
        dst_path = dst_path.replace('.webp', '.jpg')
        Image.fromarray(augmented_np).save(
            dst_path,
            format='JPEG',
            quality=95,
            subsampling=0,
            qtables='web_high'
        )
    
    def _get_augmentor(self, cls):
        if cls in ['Powdery_Mildew', 'Target_Spot']:
            try:
                crop = A.RandomResizedCrop(
                    size=Config.TARGET_SIZE,
                    scale=(0.85, 1.0),
                    ratio=(0.95, 1.05))#几何变换
            except TypeError:
                crop = A.RandomResizedCrop(
                    height=Config.TARGET_SIZE[0],
                    width=Config.TARGET_SIZE[1],
                    scale=(0.85, 1.0),
                    ratio=(0.95, 1.05))

            return A.Compose([
                crop,
                A.HorizontalFlip(p=0.3),
                A.Rotate(limit=8, p=0.2),
                A.RandomBrightnessContrast(
                    brightness_limit=0.08,
                    contrast_limit=0.08,
                    p=0.4
                ),#颜色调整
                A.CLAHE(clip_limit=2.0, p=0.3),
                # 修正后的RandomShadow参数
                A.RandomShadow(
                    shadow_roi=(0, 0.8, 1, 1),
                    shadow_dimension=5,
                    num_shadows_lower=1,
                    num_shadows_upper=2,
                    p=0.2
                ),#自然效果模拟
                # 修正后的RandomSunFlare参数
                A.RandomSunFlare(
                    flare_roi=(0, 0, 1, 0.5),
                    p=0.1
                ),
                # 修正后的Downscale参数, 暂时注释掉以避免警告
                # A.Downscale(
                #     scale_min=0.8,
                #     scale_max=0.95,
                #     interpolation=1,
                #     p=0.1
                # ),
                A.ISONoise(
                    color_shift=(0.01, 0.05),
                    intensity=(0.1, 0.3),
                    p=0.2
                ),#传感器噪声
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0
                ),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(
                    height=Config.TARGET_SIZE[0],
                    width=Config.TARGET_SIZE[1]
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0
                ),
                ToTensorV2()
            ])
    
    def _split_dataset(self, df):
        """修正索引问题的数据集划分"""
        # 确保索引连续
        df = df.reset_index(drop=True)
        
        splitter = StratifiedShuffleSplit(n_splits=1, 
                                        test_size=Config.SPLIT_RATIO[2],
                                        random_state=Config.SEED)
        
        # 生成全局有效索引
        train_val_idx, test_idx = next(splitter.split(np.zeros(len(df)), df['class']))
        
        # 使用iloc代替loc确保位置索引
        df.iloc[test_idx, df.columns.get_loc('split')] = 'test'
        
        remaining = df.iloc[train_val_idx]
        splitter = StratifiedShuffleSplit(n_splits=1,
                                        test_size=Config.SPLIT_RATIO[1]/(1-Config.SPLIT_RATIO[2]),
                                        random_state=Config.SEED)
        train_idx, val_idx = next(splitter.split(np.zeros(len(remaining)), remaining['class']))
        
        df.iloc[remaining.index[train_idx], df.columns.get_loc('split')] = 'train'
        df.iloc[remaining.index[val_idx], df.columns.get_loc('split')] = 'val'
        
        return df

class PlantDataset(Dataset):
    def __init__(self, meta_df, mode='train'):
        self.meta = meta_df
        self.mode = mode
        self.augmentor = self._init_augmentor()
        self.class_map = {cls: idx for idx, cls in enumerate(meta_df['class'].unique())}
        
    # 修改PlantDataset类的增强器
    def _init_augmentor(self):
        if self.mode == 'train':
            # 版本检测
            try:
                crop = A.RandomResizedCrop(
                    size=Config.TARGET_SIZE,
                    scale=Config.TRAIN_SCALE,
                    ratio=(0.75, 1.33))
            except TypeError:
                crop = A.RandomResizedCrop(
                    height=Config.TARGET_SIZE[0],
                    width=Config.TARGET_SIZE[1],
                    scale=(0.8, 1.2),
                    ratio=(0.75, 1.33))
            
            return A.Compose([
                crop,
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(
                    height=Config.TARGET_SIZE[0],
                    width=Config.TARGET_SIZE[1]
                ),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, idx):
        record = self.meta.iloc[idx]
        img_path = record['aug_path'] if record['is_augmented'] else record['orig_path']
        full_path = os.path.join(Config.DATA_ROOT, img_path)
        
        image = Image.open(full_path).convert('RGB')
        image = self.augmentor(image=np.array(image))['image']
        label = self.class_map[record['class']]
        
        return image, label

# 使用示例
if __name__ == "__main__":
    processor = DataProcessor()
    master_df = processor.generate_metadata()
    
    train_df = master_df[master_df['split'] == 'train']
    val_df = master_df[master_df['split'] == 'val']
    test_df = master_df[master_df['split'] == 'test']
    
    train_set = PlantDataset(train_df, mode='train')
    val_set = PlantDataset(val_df, mode='val')
    test_set = PlantDataset(test_df, mode='test')
    
    class_weights = 1.0 / train_df['class'].value_counts(normalize=True)[train_df['class']].values
    sampler = WeightedRandomSampler(
        weights=class_weights,
        num_samples=len(class_weights),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_set,
        batch_size=32,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print("Train distribution:\n", train_df['class'].value_counts())
    print("\nValidation distribution:\n", val_df['class'].value_counts())
    print("\nTest distribution:\n", test_df['class'].value_counts())

