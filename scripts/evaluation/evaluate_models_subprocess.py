#!/usr/bin/env python3
"""
子进程TFLite模型转换与评估脚本
通过子进程隔离避免TensorFlow段错误
用于生成论文Table5.5 - TFLite转换后模型对比数据
"""

import os
import time
import subprocess
import json
import tempfile
import torch
import torch.nn as nn
import torchvision.models as tv_models
import timm
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from tabulate import tabulate
from data_process import DataProcessor, PlantDataset, Config
from attention_module import CoordAtt

# 配置参数
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 1  # 模拟移动端单张推理
NUM_WORKERS = 4
IMAGE_SIZE = (224, 224)

# 模型检查点路径
CHECKPOINTS = {
    "MobileNetV3_small": "./checkpoints/MobileNetV3/best_checkpoint.pth",
    "EfficientNet_B0": "./checkpoints/efficientnetb0/best_checkpoint.pth", 
    "MobileNetV1": "./checkpoints/MobileNetV1/best_checkpoint.pth",
    "ShuffleNetV2_x1_0": "./checkpoints/shufflenetV2/best_checkpoint.pth",
    "MobileNetV3_Attention":"./checkpoints/MobileNetV3_Attention/best_checkpoint.pth",
}

# TFLite输出目录
TFLITE_DIR = "./tflite_models"
os.makedirs(TFLITE_DIR, exist_ok=True)

class MobileNetV3WithAttention(nn.Module):
    """带注意力机制的MobileNetV3模型"""
    def __init__(self, num_classes=11, attention_channels=24):
        super(MobileNetV3WithAttention, self).__init__()
        self.mobilenet = tv_models.mobilenet_v3_small(weights=None)
        self.features = self.mobilenet.features
        self.attention = CoordAtt(inp=attention_channels, oup=attention_channels)
        self.avgpool = self.mobilenet.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features[:4](x)
        x = self.attention(x)
        x = self.features[4:](x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def load_pytorch_model(name, num_classes):
    """加载PyTorch模型"""
    if name == "MobileNetV3_small":
        model = tv_models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(1024, num_classes)
    elif name == "EfficientNet_B0":
        model = tv_models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
    elif name == "MobileNetV1":
        model = timm.create_model('mobilenetv1_100', pretrained=False)
        in_features = model.get_classifier().in_features
        model.classifier = nn.Linear(in_features, num_classes)
    elif name == "ShuffleNetV2_x1_0":
        model = tv_models.shufflenet_v2_x1_0(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif name == "MobileNetV3_Attention":
        # 使用自定义带注意力机制模型
        model = MobileNetV3WithAttention(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")
    
    return model

def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_model_size_mb(model):
    """估算模型大小（MB）"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    model_size = param_size + buffer_size
    return model_size / (1024 * 1024)

def get_checkpoint_size_mb(checkpoint_path):
    """获取检查点文件大小"""
    if os.path.exists(checkpoint_path):
        return os.path.getsize(checkpoint_path) / (1024 * 1024)
    return 0.0

def measure_inference_time(model, test_loader, num_samples=100):
    """测量推理时间"""
    model.eval()
    inference_times = []
    sample_count = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            if sample_count >= num_samples:
                break
                
            images = images.to(DEVICE)
            batch_size = images.shape[0]
            
            # 逐张测量（模拟移动端场景）
            for i in range(batch_size):
                if sample_count >= num_samples:
                    break
                    
                single_image = images[i:i+1]
                
                # 预热
                for _ in range(3):
                    _ = model(single_image)
                
                # 实际测量
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                _ = model(single_image)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                
                inference_times.append((end_time - start_time) * 1000)  # 转换为毫秒
                sample_count += 1
    
    return {
        'mean_ms': np.mean(inference_times),
        'std_ms': np.std(inference_times),
        'min_ms': np.min(inference_times),
        'max_ms': np.max(inference_times),
        'samples': len(inference_times)
    }

def create_tflite_converter_script():
    """创建独立的TFLite转换脚本"""
    script_content = '''#!/usr/bin/env python3
import os
import sys
import json
import warnings
import time

# 设置环境变量
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

def safe_import_tf():
    try:
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.set_visible_devices([], 'GPU')
        return tf
    except Exception as e:
        print(f"TensorFlow import failed: {e}")
        return None

def convert_onnx_to_tflite(onnx_path, tflite_path):
    tf = safe_import_tf()
    if tf is None:
        return False, "TensorFlow not available"
    
    try:
        import onnx
        from onnx_tf.backend import prepare
        
        # 1. 验证ONNX模型有效性
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # 2. 创建临时目录
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            savedmodel_path = os.path.join(tmpdir, "saved_model")
            
            # 3. 使用 onnx-tf 将ONNX转换为SavedModel
            tf_rep = prepare(onnx_model)
            tf_rep.export_graph(savedmodel_path)
            
            # 4. 转换为TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_path)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS
            ]
            tflite_model = converter.convert()
            
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
        
        return True, f"Success: {os.path.getsize(tflite_path)} bytes"
        
    except Exception as e:
        return False, f"Conversion failed: {str(e)}"

def _preprocess_image_rgb(image_path, size=(224, 224)):
    from PIL import Image
    import numpy as np
    img = Image.open(image_path).convert('RGB')
    img = img.resize(size, resample=Image.BILINEAR)
    arr = np.asarray(img).astype('float32') / 255.0  # HWC, [0,1]
    # Normalize with ImageNet mean/std
    mean = np.array([0.485, 0.456, 0.406], dtype='float32')
    std = np.array([0.229, 0.224, 0.225], dtype='float32')
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    arr = arr.reshape(1, 3, size[0], size[1])
    return arr.astype('float32')


def _wilson_interval(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + (z*z)/n
    center = (p + (z*z)/(2*n)) / denom
    margin = (z * ((p*(1-p)/n + (z*z)/(4*n*n)) ** 0.5)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def evaluate_tflite_model(tflite_path, test_index_path):
    tf = safe_import_tf()
    if tf is None:
        return None, "TensorFlow not available"
    
    try:
        import numpy as np
        with open(test_index_path, 'r') as f:
            index = json.load(f)
        image_paths = index['image_paths']
        true_labels = index['labels']
        
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        predictions = []
        times = []
        
        for i, img_path in enumerate(image_paths):
            input_data = _preprocess_image_rgb(img_path)
            # 根据量化类型调整输入
            if input_details[0]['dtype'] == np.uint8:
                scale, zero_point = input_details[0]['quantization']
                if scale != 0:
                    input_data = input_data / scale + zero_point
                input_data = np.clip(input_data, 0, 255).astype(np.uint8)
            else:
                input_data = input_data.astype(np.float32)
            
            interpreter.set_tensor(input_details[0]['index'], input_data)
            start = time.time()
            interpreter.invoke()
            elapsed = time.time() - start
            times.append(elapsed)
            output = interpreter.get_tensor(output_details[0]['index'])
            predictions.append(int(np.argmax(output, axis=1)[0]))
        
        # 统计
        correct = int(np.sum(np.array(predictions) == np.array(true_labels)))
        n = len(true_labels)
        accuracy = correct / n if n > 0 else 0.0
        ci_low, ci_high = _wilson_interval(correct, n, z=1.96)
        avg_ms = float(np.mean(times) * 1000.0) if len(times) > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'accuracy_ci_low': ci_low,
            'accuracy_ci_high': ci_high,
            'avg_inference_time_ms': avg_ms,
            'throughput_fps': 1000.0 / avg_ms if avg_ms > 0 else 0.0,
            'total_samples': n
        }, None
        
    except Exception as e:
        return None, str(e)

if __name__ == "__main__":
    import time
    
    command = sys.argv[1]
    
    if command == "convert":
        onnx_path = sys.argv[2]
        tflite_path = sys.argv[3]
        success, message = convert_onnx_to_tflite(onnx_path, tflite_path)
        result = {"success": success, "message": message}
        print(json.dumps(result))
        
    elif command == "evaluate":
        tflite_path = sys.argv[2]
        test_index_path = sys.argv[3]
        result, error = evaluate_tflite_model(tflite_path, test_index_path)
        if result:
            print(json.dumps({"success": True, "result": result}))
        else:
            print(json.dumps({"success": False, "error": error}))
'''
    
    script_path = os.path.join(TFLITE_DIR, "tflite_converter.py")
    with open(script_path, 'w') as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)
    return script_path

def convert_to_tflite_subprocess(pytorch_model, model_name, test_loader, use_full_testset=True):
    """通过子进程安全地转换和评估TFLite模型"""
    converter_script = create_tflite_converter_script()
    
    # 1. 先导出ONNX（在主进程中，这部分是安全的）
    # 确保模型和输入在同一设备上
    pytorch_model.eval()
    
    # 将模型移到CPU进行ONNX导出（避免设备不匹配）
    model_device = next(pytorch_model.parameters()).device
    pytorch_model_cpu = pytorch_model.cpu()
    dummy_input = torch.randn(1, 3, 224, 224)  # CPU tensor
    onnx_path = os.path.join(TFLITE_DIR, f"{model_name}.onnx")
    
    try:
        print(f"🔄 导出 {model_name} 为ONNX格式...")
        torch.onnx.export(
            pytorch_model_cpu,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"✅ {model_name} ONNX导出成功")
        
        # 将模型移回原设备
        pytorch_model.to(model_device)
    except Exception as e:
        print(f"❌ ONNX导出失败: {e}")
        # 确保模型移回原设备
        pytorch_model.to(model_device)
        return None, None
    
    # 2. 准备测试数据（用于TFLite评估）
    sample_count = 0
    
    if use_full_testset:
        # 使用全测试集
        max_samples = float('inf')  # 不限制样本数量
        print(f"📊 准备全测试集数据用于TFLite评估...")
    else:
        # 仅使用50个样本（向后兼容）
        max_samples = 50
        print(f"📊 准备{max_samples}个样本用于TFLite评估...")
    
    # 从Dataset的meta构造绝对路径与标签索引，避免将整幅图像序列化到磁盘
    dataset = test_loader.dataset
    image_paths = []
    labels = []
    for idx in range(len(dataset.meta)):
        if sample_count >= max_samples:
            break
        rec = dataset.meta.iloc[idx]
        rel_path = rec['aug_path'] if rec['is_augmented'] else rec['orig_path']
        full_path = os.path.join(Config.DATA_ROOT, rel_path)
        image_paths.append(full_path)
        labels.append(int(dataset.class_map[rec['class']]))
        sample_count += 1
    
    print(f"📊 实际准备了{sample_count}个测试样本用于TFLite评估")
    
    test_index = {
        'image_paths': image_paths,
        'labels': labels
    }
    test_index_path = os.path.join(TFLITE_DIR, f"{model_name}_test_index.json")
    with open(test_index_path, 'w') as f:
        json.dump(test_index, f)
    
    # 3. 通过子进程转换ONNX到TFLite
    tflite_path = os.path.join(TFLITE_DIR, f"{model_name}.tflite")
    
    try:
        print(f"🔄 通过子进程转换 {model_name} 为TFLite...")
        result = subprocess.run([
            'python', converter_script, 'convert', onnx_path, tflite_path
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            try:
                out = result.stdout.strip()
                last_line = out.splitlines()[-1] if '\n' in out else out
                convert_result = json.loads(last_line)
                if convert_result['success']:
                    print(f"✅ {model_name} TFLite转换成功")
                    tflite_size_mb = os.path.getsize(tflite_path) / (1024 * 1024)
                else:
                    print(f"❌ {model_name} TFLite转换失败: {convert_result['message']}")
                    return None, None
            except:
                print(f"❌ {model_name} 转换结果解析失败")
                return None, None
        else:
            print(f"❌ {model_name} 转换子进程失败: {result.stderr}")
            return None, None
    except subprocess.TimeoutExpired:
        print(f"❌ {model_name} 转换超时")
        return None, None
    except Exception as e:
        print(f"❌ {model_name} 转换异常: {e}")
        return None, None
    
    # 4. 通过子进程评估TFLite模型
    try:
        print(f"📊 通过子进程评估 {model_name} TFLite模型...")
        result = subprocess.run([
            'python', converter_script, 'evaluate', tflite_path, test_index_path
        ], capture_output=True, text=True, timeout=7200)  # 全测试集评估，最大允许2小时
        
        if result.returncode == 0:
            try:
                out = result.stdout.strip()
                last_line = out.splitlines()[-1] if '\n' in out else out
                eval_result = json.loads(last_line)
                if eval_result['success']:
                    print(f"✅ {model_name} TFLite评估成功")
                    return eval_result['result'], tflite_size_mb
                else:
                    print(f"❌ {model_name} TFLite评估失败: {eval_result['error']}")
                    return None, tflite_size_mb
            except:
                print(f"❌ {model_name} 评估结果解析失败")
                return None, tflite_size_mb
        else:
            print(f"❌ {model_name} 评估子进程失败: {result.stderr}")
            return None, tflite_size_mb
    except subprocess.TimeoutExpired:
        print(f"❌ {model_name} 评估超时")
        return None, tflite_size_mb
    except Exception as e:
        print(f"❌ {model_name} 评估异常: {e}")
        return None, tflite_size_mb

def evaluate_model_accuracy(model, checkpoint_path, test_loader):
    """评估模型精度"""
    # 加载检查点
    if not os.path.exists(checkpoint_path):
        print(f"检查点文件不存在: {checkpoint_path}")
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # 尝试不同的状态字典键名
    if 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # 假设整个checkpoint就是state_dict
        model.load_state_dict(checkpoint)
    
    model = model.to(DEVICE)
    model.eval()
    
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            pred = outputs.argmax(1).cpu().numpy()
            
            predictions.extend(pred)
            true_labels.extend(labels.numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    
    # 获取检查点中的训练信息
    best_val_acc = checkpoint.get('best_acc', checkpoint.get('best', accuracy))
    epochs = checkpoint.get('epoch', 'Unknown')
    
    return {
        'test_accuracy': accuracy,
        'validation_accuracy': best_val_acc,
        'epochs_trained': epochs,
        'total_samples': len(predictions)
    }

def get_model_size_mb(path):
    return os.path.getsize(path) / (1024 * 1024) if path and os.path.exists(path) else 0.0

def main():
    print("开始模型评估...")
    print("="*80)
    
    # 准备测试数据
    processor = DataProcessor()
    master_df = processor.generate_metadata()
    test_df = master_df[master_df['split'] == 'test']
    test_dataset = PlantDataset(test_df, mode='val')  # 使用val模式确保一致预处理
    
    # 创建单张推理的数据加载器（模拟移动端）
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS, 
        pin_memory=True
    )
    
    num_classes = len(test_dataset.class_map)
    print(f"测试集样本数: {len(test_dataset)}")
    print(f"类别数: {num_classes}")
    print(f"类别映射: {test_dataset.class_map}")
    print()
    
    # 结果存储
    results = {}
    
    for model_name, checkpoint_path in CHECKPOINTS.items():
        print(f"评估模型: {model_name}")
        print("-" * 40)
        
        if not os.path.exists(checkpoint_path):
            print(f"⚠️  检查点文件不存在: {checkpoint_path}")
            print()
            continue
        
        try:
            # 加载模型
            model = load_pytorch_model(model_name, num_classes)
            
            # 评估精度
            print("📊 评估精度...")
            accuracy_results = evaluate_model_accuracy(model, checkpoint_path, test_loader)
            
            if accuracy_results is None:
                print("❌ 精度评估失败")
                continue
            
            # 计算模型统计信息
            param_count = count_parameters(model)
            model_size_mb = estimate_model_size_mb(model)
            checkpoint_size_mb = get_checkpoint_size_mb(checkpoint_path)
            
            # 测量推理时间
            print("⏱️  测量推理时间...")
            timing_results = measure_inference_time(model, test_loader, num_samples=50)
            
            # 汇总PyTorch结果
            pytorch_metrics = {
                'test_accuracy': accuracy_results['test_accuracy'],
                'validation_accuracy': accuracy_results['validation_accuracy'],
                'epochs_trained': accuracy_results['epochs_trained'],
                'param_count': param_count,
                'model_size_mb': model_size_mb,
                'checkpoint_size_mb': checkpoint_size_mb,
                'inference_time': timing_results
            }
            
            # 子进程TFLite转换与评估
            tflite_metrics, tflite_size_mb = convert_to_tflite_subprocess(model, model_name, test_loader)
            
            results[model_name] = {
                'pytorch': pytorch_metrics,
                'tflite': tflite_metrics,
                'tflite_size_mb': tflite_size_mb or 0.0
            }
            
            print(f"✅ {model_name} 评估完成")
            print(f"   测试精度: {accuracy_results['test_accuracy']:.4f}")
            print(f"   参数量: {param_count:,}")
            print(f"   推理时间: {timing_results['mean_ms']:.2f}ms")
            print()
            
        except Exception as e:
            print(f"❌ 处理模型 {model_name} 时发生错误: {e}")
            import traceback
            traceback.print_exc()
            print()
            continue
    
    # 生成对比表格
    if not results:
        print("⚠️  没有成功评估的模型")
        return
    
    print("\n" + "="*100)
    print("Table 5.5 - TFLite转换后模型对比")
    print("="*100)
    
    headers = [
        "模型", 
        "原始精度(%)", "TFLite精度(%)", "精度损失(%)",
        "原始大小(MB)", "TFLite大小(MB)", "压缩比",
        "原始延迟(ms)", "TFLite延迟(ms)", "加速比",
        "参数量(M)"
    ]
    
    table_data = []
    
    for model_name, result in results.items():
        pytorch_res = result['pytorch']
        tflite_res = result['tflite']
        
        pytorch_acc = pytorch_res['test_accuracy'] * 100
        pytorch_size = pytorch_res['checkpoint_size_mb']
        pytorch_latency = pytorch_res['inference_time']['mean_ms']
        param_count_m = pytorch_res['param_count'] / 1e6

        if tflite_res:
            tflite_acc = tflite_res['accuracy'] * 100
            acc_loss = pytorch_acc - tflite_acc
            tflite_size = result.get('tflite_size_mb', 0.0)
            compression_ratio = (pytorch_size / tflite_size) if tflite_size > 0 else float('inf')
            tflite_latency = tflite_res['avg_inference_time_ms']
            speedup = pytorch_latency / tflite_latency if tflite_latency > 0 else 0.0

            table_data.append([
                model_name,
                f"{pytorch_acc:.2f}",
                f"{tflite_acc:.2f}",
                f"{acc_loss:.2f}",
                f"{pytorch_size:.2f}",
                f"{tflite_size:.2f}",
                f"{compression_ratio:.1f}x" if compression_ratio != float('inf') else "N/A",
                f"{pytorch_latency:.2f}",
                f"{tflite_latency:.2f}",
                f"{speedup:.1f}x" if speedup > 0 else "N/A",
                f"{param_count_m:.2f}"
            ])
        else:
            table_data.append([
                model_name,
                f"{pytorch_acc:.2f}",
                "转换失败",
                "N/A",
                f"{pytorch_size:.2f}",
                "N/A",
                "N/A",
                f"{pytorch_latency:.2f}",
                "N/A",
                "N/A",
                f"{param_count_m:.2f}"
            ])

    print(tabulate(table_data, headers=headers, tablefmt='grid'))

    # 保存详细结果
    print("\n" + "="*60)
    print("详细性能指标")
    print("="*60)

    for model_name, result in results.items():
        pytorch_res = result['pytorch']
        tflite_res = result['tflite']

        print(f"\n{model_name}:")
        print(f"  原始模型:")
        print(f"    测试精度: {pytorch_res['test_accuracy']:.4f}")
        print(f"    验证精度: {pytorch_res['validation_accuracy']:.4f}")
        print(f"    参数量: {pytorch_res['param_count']:,}")
        print(f"    模型大小: {pytorch_res['checkpoint_size_mb']:.2f} MB")
        print(f"    推理时间: {pytorch_res['inference_time']['mean_ms']:.2f} ± {pytorch_res['inference_time']['std_ms']:.2f} ms")
        
        if tflite_res:
            print(f"  TFLite模型:")
            print(f"    精度: {tflite_res['accuracy']:.4f}")
            print(f"    大小: {result['tflite_size_mb']:.2f} MB")
            print(f"    推理时间: {tflite_res['avg_inference_time_ms']:.2f} ms")
            print(f"    吞吐量: {tflite_res['throughput_fps']:.1f} FPS")
            print(f"    样本数: {tflite_res['total_samples']}")
        else:
            print(f"  TFLite模型: 转换失败")

    # 保存CSV结果
    csv_data = []
    for model_name, result in results.items():
        pytorch_res = result['pytorch']
        tflite_res = result['tflite']
        
        row = {
            'model_name': model_name,
            'pytorch_test_accuracy': pytorch_res['test_accuracy'],
            'pytorch_validation_accuracy': pytorch_res['validation_accuracy'],
            'pytorch_param_count': pytorch_res['param_count'],
            'pytorch_size_mb': pytorch_res['checkpoint_size_mb'],
            'pytorch_inference_time_ms': pytorch_res['inference_time']['mean_ms'],
            'pytorch_inference_std_ms': pytorch_res['inference_time']['std_ms'],
            'epochs_trained': pytorch_res['epochs_trained']
        }
        
        if tflite_res:
            row.update({
                'tflite_accuracy': tflite_res['accuracy'],
                'tflite_accuracy_ci_low': tflite_res.get('accuracy_ci_low'),
                'tflite_accuracy_ci_high': tflite_res.get('accuracy_ci_high'),
                'tflite_size_mb': result['tflite_size_mb'],
                'tflite_inference_time_ms': tflite_res['avg_inference_time_ms'],
                'tflite_throughput_fps': tflite_res['throughput_fps'],
                'tflite_samples': tflite_res['total_samples'],
                'accuracy_loss': pytorch_res['test_accuracy'] - tflite_res['accuracy'],
                'compression_ratio': pytorch_res['checkpoint_size_mb'] / result['tflite_size_mb'] if result['tflite_size_mb'] > 0 else 0,
                'speedup_ratio': pytorch_res['inference_time']['mean_ms'] / tflite_res['avg_inference_time_ms'] if tflite_res['avg_inference_time_ms'] > 0 else 0
            })
        else:
            row.update({
                'tflite_accuracy': None,
                'tflite_size_mb': None,
                'tflite_inference_time_ms': None,
                'tflite_throughput_fps': None,
                'tflite_samples': None,
                'accuracy_loss': None,
                'compression_ratio': None,
                'speedup_ratio': None
            })
        
        csv_data.append(row)
    
    df = pd.DataFrame(csv_data)
    df.to_csv('model_tflite_comparison_results.csv', index=False)
    print(f"\n💾 详细结果已保存到: model_tflite_comparison_results.csv")

if __name__ == "__main__":
    main()