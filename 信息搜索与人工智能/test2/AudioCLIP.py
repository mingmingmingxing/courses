import os
import sys
import glob

import librosa
import librosa.display

import simplejpeg
import numpy as np

import torch
import torchvision as tv

import matplotlib.pyplot as plt

from PIL import Image
from IPython.display import Audio, display

sys.path.append(os.path.abspath(f'{os.getcwd()}'))
# sys.path.append("/home/shuchenweng/cz/oyh/others/AudioCLIP")

from model import AudioCLIP
from utils.transforms import ToTensor1D


torch.set_grad_enabled(False)

MODEL_FILENAME = 'AudioCLIP-Full-Training.pt'
# derived from ESResNeXt
SAMPLE_RATE = 44100
# derived from CLIP
IMAGE_SIZE = 224
IMAGE_MEAN = 0.48145466, 0.4578275, 0.40821073
IMAGE_STD = 0.26862954, 0.26130258, 0.27577711


LABELS = ['cat', 'thunderstorm', 'coughing', 'alarm clock', 'car horn']

##### 初始化模型和加载权重
aclp = AudioCLIP(pretrained=f'assets/{MODEL_FILENAME}').to('cpu')

##### 图像和音频数据预处理，对图像随机裁剪到指定大小
audio_transforms = ToTensor1D()

image_transforms = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Resize(IMAGE_SIZE, interpolation=Image.BICUBIC),
    tv.transforms.CenterCrop(IMAGE_SIZE),
    tv.transforms.Normalize(IMAGE_MEAN, IMAGE_STD)
])

##### 音频文件的加载
paths_to_audio = glob.glob('./demo/audio/*.wav')
audio = list()
for path_to_audio in paths_to_audio:
    track, _ = librosa.load(path_to_audio, sr=SAMPLE_RATE, dtype=np.float32)
    # compute spectrograms using trained audio-head (fbsp-layer of ESResNeXt)
    # thus, the actual time-frequency representation will be visualized
    spec = aclp.audio.spectrogram(torch.from_numpy(track.reshape(1, 1, -1)))
    spec = np.ascontiguousarray(spec.numpy()).view(np.complex64)
    pow_spec = 10 * np.log10(np.abs(spec) ** 2 + 1e-18).squeeze()
    audio.append((track, pow_spec))

#### 音频文件可视化
fig, axes = plt.subplots(2, len(audio), figsize=(20, 5), dpi=100)
for idx in range(len(audio)):
    track, pow_spec = audio[idx]
    # draw the waveform
    librosa.display.waveplot(track, sr=SAMPLE_RATE, ax=axes[0, idx], color='k')
    # show the corresponding power spectrogram
    axes[1, idx].imshow(pow_spec, origin='lower', aspect='auto', cmap='gray', vmin=-180.0, vmax=20.0)
    # modify legend
    axes[0, idx].set_title(os.path.basename(paths_to_audio[idx]))
    axes[0, idx].set_xlabel('')
    axes[0, idx].set_xticklabels([])
    axes[0, idx].grid(True)
    axes[0, idx].set_ylim(bottom=-1, top=1)
    axes[1, idx].set_xlabel('Time (s)')
    axes[1, idx].set_xticks(np.linspace(0, pow_spec.shape[1], len(axes[0, idx].get_xticks())))
    axes[1, idx].set_xticklabels([f'{tick:.1f}' if tick == int(tick) else '' for tick in axes[0, idx].get_xticks()])
    axes[1, idx].set_yticks(np.linspace(0, pow_spec.shape[0] - 1, 5))
axes[0, 0].set_ylabel('Amplitude')
axes[1, 0].set_ylabel('Filter ID')
plt.show()
plt.close(fig)
for idx, path in enumerate(paths_to_audio):
    print(os.path.basename(path))
    display(Audio(audio[idx][0], rate=SAMPLE_RATE, embed=True))

#### 图像文件的加载
paths_to_images = glob.glob('./demo/images/*.jpg')
images = list()
for path_to_image in paths_to_images:
    with open(path_to_image, 'rb') as jpg:
        image = simplejpeg.decode_jpeg(jpg.read())
        images.append(image)

#### 图像文件可视化
fig, axes = plt.subplots(2, len(images) // 2, figsize=(16, 4), dpi=100)
for idx, jdx in np.ndindex(axes.shape):
    # re-arrange order to show the images column-wise
    image_idx = np.ravel_multi_index(((jdx,), (idx,)), axes.shape[::-1]).item()
    axes[idx, jdx].imshow(images[image_idx])
    # modify legend
    axes[idx, jdx].axis('off')
    axes[idx, jdx].set_title(os.path.basename(paths_to_images[image_idx]))
plt.show()
plt.close(fig)

# AudioCLIP 处理输入的原始音频，因此输入形状为 [batch x channels x duration]
audio = torch.stack([audio_transforms(track.reshape(1, -1)) for track, _ in audio])
# 标准通道优先形状 [batch x channels x height x width]
images = torch.stack([image_transforms(image) for image in images])
# 文本输入在内部处理，因此无需事先进行转换
text = [[label] for label in LABELS]

# AudioCLIP's output: Tuple[Tuple[Features, Logits], Loss]
# Features = Tuple[AudioFeatures, ImageFeatures, TextFeatures]
# Logits = Tuple[AudioImageLogits, AudioTextLogits, ImageTextLogits]
# 获取 文本，图像，音频 Embeddings
((audio_features, _, _), _), _ = aclp(audio=audio)
((_, image_features, _), _), _ = aclp(image=images)
((_, _, text_features), _), _ = aclp(text=text)

audio_features = audio_features / torch.linalg.norm(audio_features, dim=-1, keepdim=True)
image_features = image_features / torch.linalg.norm(image_features, dim=-1, keepdim=True)
text_features = text_features / torch.linalg.norm(text_features, dim=-1, keepdim=True)

scale_audio_image = torch.clamp(aclp.logit_scale_ai.exp(), min=1.0, max=100.0)
scale_audio_text = torch.clamp(aclp.logit_scale_at.exp(), min=1.0, max=100.0)
scale_image_text = torch.clamp(aclp.logit_scale.exp(), min=1.0, max=100.0)

# 计算相似度
logits_audio_image = scale_audio_image * audio_features @ image_features.T
logits_audio_text = scale_audio_text * audio_features @ text_features.T
logits_image_text = scale_image_text * image_features @ text_features.T


print('\t\tFilename, Audio\t\t\tTextual Label (Confidence)', end='\n\n')
# calculate model confidence
confidence = logits_audio_text.softmax(dim=1)
for audio_idx in range(len(paths_to_audio)):
    # acquire Top-3 most similar results
    conf_values, ids = confidence[audio_idx].topk(3)

    # format output strings
    query = f'{os.path.basename(paths_to_audio[audio_idx]):>30s} ->\t\t'
    results = ', '.join([f'{LABELS[i]:>15s} ({v:06.2%})' for v, i in zip(conf_values, ids)])

    print(query + results)


print('\tFilename, Image\t\t\tTextual Label (Confidence)', end='\n\n')

# calculate model confidence
confidence = logits_image_text.softmax(dim=1)
for image_idx in range(len(paths_to_images)):
    # acquire Top-3 most similar results
    conf_values, ids = confidence[image_idx].topk(3)

    # format output strings
    query = f'{os.path.basename(paths_to_images[image_idx]):>20s} ->\t\t'
    results = ', '.join([f'{LABELS[i]:>20s} ({v:06.2%})' for v, i in zip(conf_values, ids)])

    print(query + results)


print('\t\tTextual Label\t\tFilename, Audio (Confidence)', end='\n\n')
# calculate model confidence
confidence = logits_audio_text.softmax(dim=0)
for label_idx in range(len(LABELS)):
    # acquire Top-2 most similar results
    conf_values, ids = confidence[:, label_idx].topk(2)
    # format output strings
    query = f'{LABELS[label_idx]:>25s} ->\t\t'
    results = ', '.join([f'{os.path.basename(paths_to_audio[i]):>30s} ({v:06.2%})' for v, i in zip(conf_values, ids)])
    print(query + results)

### 音频与文本的检索
print('\tTextual Label\t\t\tFilename, Image (Confidence)', end='\n\n')

# calculate model confidence
confidence = logits_image_text.softmax(dim=0)
for label_idx in range(len(LABELS)):
    # acquire Top-3 most similar results
    conf_values, ids = confidence[:, label_idx].topk(3)

    # format output strings
    query = f'{LABELS[label_idx]:>20s} ->\t\t'
    results = ', '.join([f'{os.path.basename(paths_to_images[i]):>20s} ({v:>06.2%})' for v, i in zip(conf_values, ids)])

    print(query + results)

### 音频与图像的检索
print('\tTextual Label\t\t\tFilename, Image (Confidence)', end='\n\n')
# calculate model confidence
confidence = logits_audio_image.softmax(dim=0)
for image_idx in range(len(paths_to_images)):
    # acquire Top-2 most similar results
    conf_values, ids = confidence[:, image_idx].topk(2)
    # format output strings
    query = f'{os.path.basename(paths_to_images[image_idx]):>25s} ->\t\t'
    results = ', '.join([f'{os.path.basename(paths_to_audio[i]):>30s} ({v:06.2%})' for v, i in zip(conf_values, ids)])

    print(query + results)

### Images by Audio
print('\tTextual Label\t\t\tFilename, Image (Confidence)', end='\n\n')

# calculate model confidence
confidence = logits_audio_image.softmax(dim=1)
for audio_idx in range(len(paths_to_audio)):
    # acquire Top-3 most similar results
    conf_values, ids = confidence[audio_idx].topk(3)

    # format output strings
    query = f'{os.path.basename(paths_to_audio[audio_idx]):>30s} ->\t\t'
    results = ', '.join([f'{os.path.basename(paths_to_images[i]):>15s} ({v:06.2%})' for v, i in zip(conf_values, ids)])

    print(query + results)