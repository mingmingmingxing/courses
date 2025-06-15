import os
import glob
import subprocess
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from tqdm import tqdm
import librosa
import matplotlib
import matplotlib.pyplot as plt
from model import AudioCLIP
from utils.transforms import ToTensor1D

matplotlib.rcParams['font.family'] = 'Microsoft YaHei'

# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

aclp = AudioCLIP(pretrained='assets/AudioCLIP-Full-Training.pt').to(device)
aclp.eval()

audio_transforms = ToTensor1D()

IMAGE_SIZE = 224
IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

image_transforms = T.Compose([
    T.ToTensor(),
    T.Resize(IMAGE_SIZE),
    T.CenterCrop(IMAGE_SIZE),
    T.Normalize(IMAGE_MEAN, IMAGE_STD)
])

SAMPLE_RATE = 44100
TOP_K = 5  # 最多检索前5张最相似的帧图像

video_dir = './video'
output_audio_dir = './demo/audio'
output_image_dir = './demo/images'
output_result_dir = './demo/results'  # 可视化图像保存路径
os.makedirs(output_audio_dir, exist_ok=True)
os.makedirs(output_image_dir, exist_ok=True)
os.makedirs(output_result_dir, exist_ok=True)

# ========== 步骤 1：提取音频和帧 ==========
print("正在提取视频音频和帧...")
video_files = glob.glob(os.path.join(video_dir, '*.mp4'))

for video_path in video_files:
    basename = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(output_audio_dir, f'{basename}.wav')
    frame_dir = os.path.join(output_image_dir, basename)
    os.makedirs(frame_dir, exist_ok=True)

    # 提取音频
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vn', '-ac', '1', '-ar', str(SAMPLE_RATE), audio_path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 提取帧（每秒1帧）
    subprocess.run(['ffmpeg', '-y', '-i', video_path, '-vf', 'fps=1', os.path.join(frame_dir, 'frame_%04d.jpg')],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # ========== 步骤 2：编码音频 ==========
    print(f"处理音频: {audio_path}")
    track, _ = librosa.load(audio_path, sr=SAMPLE_RATE, dtype=np.float32)
    audio_tensor = audio_transforms(track.reshape(1, -1)).unsqueeze(0).to(device)
    ((audio_features, _, _), _), _ = aclp(audio=audio_tensor)
    audio_features = audio_features / torch.linalg.norm(audio_features, dim=-1, keepdim=True)

    # ========== 步骤 3：编码所有帧图像 ==========
    print(f"处理帧图像: {frame_dir}")
    frame_paths = sorted(glob.glob(os.path.join(frame_dir, '*.jpg')))
    images = []
    for img_path in tqdm(frame_paths):
        image = Image.open(img_path).convert('RGB')
        image_tensor = image_transforms(image)
        images.append(image_tensor)

    images = torch.stack(images).to(device)
    ((_, image_features, _), _), _ = aclp(image=images)
    image_features = image_features / torch.linalg.norm(image_features, dim=-1, keepdim=True)

    # ========== 步骤 4：计算相似度 ==========
    similarity = (audio_features @ image_features.T).squeeze(0)
    topk_vals, topk_idxs = similarity.topk(TOP_K)

    print(f"\n 音频: {basename}.wav 匹配到的 Top-{TOP_K} 图像帧:")
    top_frames = []
    for i, idx in enumerate(topk_idxs):
        match_path = frame_paths[idx]
        score = topk_vals[i].item()
        print(f"  Top-{i+1}: {match_path} | 相似度: {score:.4f}")
        top_frames.append((match_path, score))

    print("-" * 60)

    # ========== 可视化结果 ==========
    fig, axs = plt.subplots(1, TOP_K, figsize=(18, 4))
    fig.suptitle(f'Top-{TOP_K} Frames for Audio: {basename}', fontsize=14)

    for i, (frame_path, score) in enumerate(top_frames):
        img = Image.open(frame_path).convert('RGB')
        axs[i].imshow(img)
        axs[i].axis('off')
        axs[i].set_title(f'{os.path.basename(frame_path)}\nScore: {score:.4f}', fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.75)  # 给标题留空间
    save_path = os.path.join(output_result_dir, f'{basename}_top{TOP_K}.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f" Top-{TOP_K} 可视化结果已保存至：{save_path}")
