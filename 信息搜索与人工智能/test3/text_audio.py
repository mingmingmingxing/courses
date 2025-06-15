import os
import glob
import torch
import numpy as np
from pydub import AudioSegment
from model import AudioCLIP
from utils.transforms import ToTensor1D

# Configuration
SAMPLE_RATE = 44100
TARGET_LENGTH = 441000  # 10秒音频长度
BATCH_SIZE = 8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

MODEL_FILENAME = 'AudioCLIP-Full-Training.pt'
aclp = AudioCLIP(pretrained=f'assets/{MODEL_FILENAME}').to(device)
audio_transforms = ToTensor1D()

# Define the labels for retrieval
LABELS = [
    'joy', 'anger', 'sadness', 'fear', 'surprise', 'anxiety', 'anticipation', 'guilt', 'contentment',
    'confusion', 'loneliness', 'excitement', 'calmness', 'envy', 'compassion', 'pride', 'disgust', 'helplessness',
    'tension', 'relaxation', 'irritation', 'curiosity', 'disappointment', 'gratitude', 'admiration', 'pity',
    'oppression', 'elation', 'melancholy', 'ecstasy', 'indifference'
]

def load_and_fix_audio(path, target_length=TARGET_LENGTH):
    """Robustly load and normalize audio using pydub"""
    try:
        audio = AudioSegment.from_file(path)
        audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
        samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0

        if len(samples) > target_length:
            samples = samples[:target_length]
        else:
            samples = np.pad(samples, (0, target_length - len(samples)), mode='constant')

        return samples
    except Exception as e:
        print(f"[!] Error loading {os.path.basename(path)}: {str(e)}")
        return None

def load_audio_files(audio_dir):
    """Load all audio files from directory"""
    paths = glob.glob(os.path.join(audio_dir, '*.mp3')) + glob.glob(os.path.join(audio_dir, '*.wav'))
    print(f"[+] Found {len(paths)} audio files in {audio_dir}")

    audio_data = []
    for path in paths:
        track = load_and_fix_audio(path)
        if track is not None:
            audio_data.append((path, track))

    return audio_data

def extract_features(audio_data, batch_size=BATCH_SIZE):
    """Extract AudioCLIP features in batches to avoid memory overflow"""
    paths = []
    all_features = []
    with torch.no_grad():
        for i in range(0, len(audio_data), batch_size):
            batch = audio_data[i:i + batch_size]
            batch_paths = [item[0] for item in batch]
            batch_tracks = [item[1] for item in batch]
            audio_batch = torch.stack([audio_transforms(track.reshape(1, -1)) for track in batch_tracks]).to(device)
            ((audio_features, _, _), _), _ = aclp(audio=audio_batch)
            audio_features = audio_features / torch.linalg.norm(audio_features, dim=-1, keepdim=True)
            paths.extend(batch_paths)
            all_features.append(audio_features.cpu())  # offload to CPU
    return paths, torch.cat(all_features, dim=0)  # Returns on CPU

def text_to_audio_retrieval(text_labels, audio_paths, audio_features, top_k=3):
    """Perform text-to-audio retrieval"""
    with torch.no_grad():
        text = [[label] for label in text_labels]
        ((_, _, text_features), _), _ = aclp(text=text)
        text_features = text_features / torch.linalg.norm(text_features, dim=-1, keepdim=True)

        audio_features = audio_features.to(text_features.device)

        scale = torch.clamp(aclp.logit_scale_at.exp(), min=1.0, max=100.0)
        logits = scale * text_features @ audio_features.T
        confidence = logits.softmax(dim=1)

        results = {}
        for label_idx, label in enumerate(text_labels):
            conf_values, indices = confidence[label_idx].topk(top_k)
            results[label] = [(audio_paths[i], float(conf_values[j])) for j, i in enumerate(indices)]

        return results

def print_results(retrieval_results):
    """Print retrieval results in a readable format"""
    print("\n[Text-to-Audio Retrieval Results]")
    print("=" * 60)
    for label, matches in retrieval_results.items():
        print(f"\nQuery: {label}")
        for i, (audio_path, confidence) in enumerate(matches, 1):
            print(f"  {i}. {os.path.basename(audio_path)} ({confidence:.2%})")

if __name__ == "__main__":
    audio_dir = "./audio"
    audio_data = load_audio_files(audio_dir)

    if not audio_data:
        print("[-] No valid audio files loaded. Please check the folder.")
        exit()

    audio_paths, audio_features = extract_features(audio_data)
    retrieval_results = text_to_audio_retrieval(LABELS, audio_paths, audio_features)
    print_results(retrieval_results)