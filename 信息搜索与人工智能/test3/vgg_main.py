# audio_text_retrieval.py
# 用 VGG16 + BERT 完成音频與文本的相互检索（本地加载模型，读取 ./audio 中的音频，包括 .mp3 格式）

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchaudio
from transformers import BertModel, BertTokenizer
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, Resample


# ========== 1. 音频编码器（VGG16） ==========
class AudioEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        vgg = models.vgg16_bn()
        vgg.load_state_dict(torch.load("./pretrained/vgg16_bn.pth"))
        self.features = vgg.features[0:23]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(256, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(0.2)  # 增加dropout率

        # 增强情感特征提取
        self.emotion_proj = nn.Sequential(
            nn.Linear(output_dim, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.2),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )
        # 添加多尺度特征提取
        self.multi_scale = nn.ModuleList([
            nn.Linear(output_dim, output_dim),
            nn.Linear(output_dim, output_dim),
            nn.Linear(output_dim, output_dim)
        ])

        self.audio_proj = nn.Linear(output_dim, 256)

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        print(f"Pool output shape: {x.shape}")

        # 基础特征
        base_feat = self.proj(x)
        base_feat = self.layer_norm(base_feat)
        base_feat = self.dropout(base_feat)

        # 情感特征
        emotion_feat = self.emotion_proj(base_feat)

        # 多尺度特征
        multi_scale_feats = []
        for layer in self.multi_scale:
            feat = layer(base_feat)
            multi_scale_feats.append(feat)

        # 融合所有特征
        x = base_feat + emotion_feat + sum(multi_scale_feats)

        # 映射到256维
        x = self.audio_proj(x)

        return F.normalize(x, dim=-1)


# ========== 2. 文本编码器（BERT） ==========
class TextEncoder(nn.Module):
    def __init__(self):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained("./pretrained/bert-base-uncased")
        self.proj = nn.Linear(768, 256)  # 从768映射到256

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = output.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size())
        masked = last_hidden * mask
        summed = torch.sum(masked, 1)
        counts = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed / counts
        projected = self.proj(mean_pooled)  # 映射到256维
        return F.normalize(projected, dim=-1)


# ========== 3. 对比损失函数 ==========
def contrastive_loss(a_feat, t_feat, temperature=0.07):
    logits = a_feat @ t_feat.T / temperature
    labels = torch.arange(a_feat.size(0)).to(a_feat.device)
    loss_a2t = F.cross_entropy(logits, labels)
    loss_t2a = F.cross_entropy(logits.T, labels)
    return (loss_a2t + loss_t2a) / 2


# ========== 4. 检索函数 ==========
def retrieve_text_from_audio(audio_vec, text_vecs, texts, top_k=4):
    audio_vec = F.normalize(audio_vec, dim=-1)
    text_vecs = F.normalize(text_vecs, dim=-1)

    sims = torch.matmul(audio_vec, text_vecs.T)
    topk_vals, topk_idxs = sims.topk(min(top_k, len(texts)))
    return [(texts[i.item()], topk_vals[0, j].item()) for j, i in enumerate(topk_idxs[0])]

def retrieve_audio_from_text(text_vec, audio_vecs, audio_files, top_k=3):
    text_vec = F.normalize(text_vec, dim=-1)
    audio_vecs = F.normalize(audio_vecs, dim=-1)

    sims = torch.matmul(text_vec, audio_vecs.T)
    topk_vals, topk_idxs = sims.topk(min(top_k, len(audio_files)))
    return [(audio_files[i.item()], topk_vals[0, j].item()) for j, i in enumerate(topk_idxs[0])]


# ========== 5. 音频预处理函数 ==========
def load_audio_mel(path):
    waveform, sr = torchaudio.load(path)  # [channels, time]
    if sr != 16000:
        waveform = Resample(sr, 16000)(waveform)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # 单通道
    mel = MelSpectrogram(sample_rate=16000, n_mels=64, n_fft=1024)(waveform)  # 调整参数
    db = AmplitudeToDB()(mel)
    db = F.interpolate(db.unsqueeze(1), size=(224, 224), mode="bilinear", align_corners=False)
    db = db.repeat(1, 3, 1, 1)
    return db


# ========== 6. 主程序 ==========
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("使用设备：", device)

    # 初始化模型
    audio_model = AudioEncoder().to(device).eval()
    text_model = TextEncoder().to(device).eval()
    tokenizer = BertTokenizer.from_pretrained("./pretrained/bert-base-uncased")

    texts = [
        "joy | I feel very happy and excited today.",  # joy
        "anger | I am filled with anger and frustration.",  # anger
        "sadness | I am feeling very sad and hopeless.",  # sadness
        "fear | I am scared and anxious about the future.",  # fear
        "peace | What a peaceful and calm moment it is.",  # peace
        "anxious | I am nervous and can't concentrate.",  # anxious
        "warm | My heart is warm and full of love.",  # warm
        "energetic | I feel energetic and ready to move!"  # energetic
    ]

    # 文本编码与归一化
    encoding = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    encoding.pop("token_type_ids", None)
    encoding = {k: v.to(device) for k, v in encoding.items()}
    text_vecs = text_model(**encoding)  # 返回归一化向量

    # 打印文本向量之间的相似度矩阵
    with torch.no_grad():
        sim_matrix = torch.matmul(text_vecs, text_vecs.T)
        print("\n🧠 文本向量之间的相似度矩阵:")
        for i in range(len(texts)):
            sims = ["{:.2f}".format(sim_matrix[i, j].item()) for j in range(len(texts))]
            print(f"{texts[i]:<40}: {'  '.join(sims)}")

    # 预处理所有音频文件
    audio_dir = './audio'
    audio_files = []
    audio_vecs = []

    print("\n🎵 预处理所有音频文件...")
    for file in os.listdir(audio_dir):
        if not (file.endswith('.wav') or file.endswith('.mp3')):
            continue
        path = os.path.join(audio_dir, file)
        print(f"处理音频: {file}")
        audio_tensor = load_audio_mel(path).to(device)
        audio_vec = audio_model(audio_tensor)
        audio_files.append(file)
        audio_vecs.append(audio_vec)
    
    if audio_vecs:
        audio_vecs = torch.cat(audio_vecs, dim=0)
        
        # 1. 音频检索文本
        print("\n🔍 音频检索文本示例:")
        for i, audio_file in enumerate(audio_files):
            print(f"\n当前音频: {audio_file}")
            results = retrieve_text_from_audio(audio_vecs[i:i+1], text_vecs, texts, top_k=5)
            for text, score in results:
                print(f"  匹配文本: {text} | 相似度: {score:.4f}")
        
        # 2. 文本检索音频
        print("\n🔍 文本检索音频示例:")
        for i, text in enumerate(texts):
            print(f"\n当前文本: {text}")
            results = retrieve_audio_from_text(text_vecs[i:i+1], audio_vecs, audio_files, top_k=5)
            for audio_file, score in results:
                print(f"  匹配音频: {audio_file} | 相似度: {score:.4f}")
    else:
        print("没有找到任何音频文件！")