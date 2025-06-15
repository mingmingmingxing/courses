import os
import sys
import glob
import pandas as pd
import numpy as np
import torch
import torchvision as tv
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import simplejpeg
from tqdm import tqdm
from collections import Counter

# 将项目根目录添加到 Python 路径
sys.path.append(os.path.abspath(f'{os.path.dirname(os.getcwd())}'))

from model import AudioCLIP
from utils.transforms import ToTensor1D

# 检查 GPU 是否可用
print("\n=== GPU 状态检查 ===")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 设备数量: {torch.cuda.device_count()}")
    print(f"当前 GPU 设备: {torch.cuda.current_device()}")
    print(f"GPU 设备名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU 内存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"GPU 内存已用: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"GPU 内存缓存: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
else:
    print("警告: 未检测到可用的 GPU，将使用 CPU 运行")

print(f"使用设备: {device}\n")

torch.set_grad_enabled(False)

MODEL_FILENAME = 'AudioCLIP-Full-Training.pt'
SAMPLE_RATE = 44100

def load_esc50_data(data_dir):
    """加载 ESC50 数据集"""
    print("正在加载 ESC50 数据集...")
    meta_file = os.path.join(data_dir, 'meta', 'esc50.csv')
    audio_dir = os.path.join(data_dir, 'audio')
    
    if not os.path.exists(meta_file):
        raise FileNotFoundError(f"找不到元数据文件: {meta_file}")
    if not os.path.exists(audio_dir):
        raise FileNotFoundError(f"找不到音频目录: {audio_dir}")
    
    # 读取元数据
    meta = pd.read_csv(meta_file)
    print(f"找到 {len(meta)} 个音频文件")
    
    # 从元数据中提取所有唯一的类别作为 LABELS
    global LABELS
    LABELS = sorted(meta['category'].unique().tolist())
    print(f"从数据集中提取 {len(LABELS)} 个类别")
    
    # 加载音频文件
    audio_files = []
    labels = []
    
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="加载音频文件"):
        audio_path = os.path.join(audio_dir, row['filename'])
        if os.path.exists(audio_path):
            audio_files.append(audio_path)
            labels.append(row['category'])
            # 打印加载的文件路径和类别
            # print(f"加载文件: {audio_path}, 类别: {row['category']}")
    
    print(f"成功加载 {len(audio_files)} 个音频文件")
    return audio_files, labels

def process_audio(audio_path, model, device):
    """处理单个音频文件"""
    # 加载音频
    track, _ = librosa.load(audio_path, sr=SAMPLE_RATE, dtype=np.float32)
    
    # 转换为模型输入格式，添加一个额外的维度作为批处理并移到设备
    audio = torch.from_numpy(track.reshape(1, -1)).unsqueeze(0).to(device)
    
    return audio

def evaluate_model(model, audio_files, true_labels, device, batch_size=16):
    """评估模型性能"""
    predictions = []
    confidences = []
    
    # 将音频文件分批处理
    for i in tqdm(range(0, len(audio_files), batch_size), desc="处理音频批次"):
        batch_files = audio_files[i:i + batch_size]
        batch_audios = []
        
        # 处理每个音频文件并移到设备
        for audio_path in batch_files:
            audio = process_audio(audio_path, model, device)
            batch_audios.append(audio)
        
        # 将批次中的音频堆叠在一起
        batch_audio = torch.cat(batch_audios, dim=0)
        
        # 获取模型预测
        ((audio_features, _, _), _), _ = model(audio=batch_audio)
        audio_features = audio_features / torch.linalg.norm(audio_features, dim=-1, keepdim=True)
        
        # 计算与所有标签的相似度
        text = [[label] for label in LABELS]
        # 将文本特征移到设备
        text_inputs = model.encode_text(text).to(device)
        text_features = text_inputs / torch.linalg.norm(text_inputs, dim=-1, keepdim=True)
        
        # 计算相似度分数
        scale_audio_text = torch.clamp(model.logit_scale_at.exp(), min=1.0, max=100.0)
        logits_audio_text = scale_audio_text * audio_features @ text_features.T
        
        # 获取预测结果
        confidence = logits_audio_text.softmax(dim=1)
        pred_indices = confidence.argmax(dim=1)
        
        # 保存预测结果
        for j in range(len(batch_files)):
            pred_idx = pred_indices[j].item()
            predictions.append(LABELS[pred_idx])
            confidences.append(confidence[j][pred_idx].item())
        
        # 清理 GPU 内存
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return predictions, confidences

def plot_confusion_matrix(y_true, y_pred, labels):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    print("\n=== ESC50 数据集评估 ===")
    try:
        # 初始化模型并移到设备
        aclp = AudioCLIP(pretrained=f'assets/{MODEL_FILENAME}').to(device)
        
        # 加载数据
        data_dir = 'ESC-50-master'  # ESC50 数据集目录
        audio_files, true_labels = load_esc50_data(data_dir)
        
        # 统计每个类别的样本数量
        label_counts = Counter(true_labels)
        print("\n加载数据集中的类别分布:")
        for label in LABELS:
            print(f"{label}: {label_counts.get(label, 0)} 样本")
        
        # 评估模型
        predictions, confidences = evaluate_model(aclp, audio_files, true_labels, device)
        
        # 打印分类报告
        print("\n分类报告:")
        print(classification_report(true_labels, predictions, labels=LABELS, zero_division=0))
        
        # 绘制混淆矩阵
        plot_confusion_matrix(true_labels, predictions, LABELS)
        
        # 打印每个类别的准确率
        print("\n各类别准确率:")
        for label in LABELS:
            mask = [l == label for l in true_labels]
            if sum(mask) > 0:
                acc = sum([p == t for p, t in zip(predictions, true_labels) if t == label]) / sum(mask)
                print(f"{label}: {acc:.2%}")
        
    except FileNotFoundError as e:
        print(f"文件未找到错误: {str(e)}")
    except Exception as e:
        print(f"发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    main() 