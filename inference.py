import torch
import torch.nn as nn
import torchaudio
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
from PIL import Image
import os
import tensorflow_hub as hub   # needed for YAMNet embeddings

# -----------------------------
# Config
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths to models
efficientnet_path = "efficientnet_birds.pth"
resnet_path = "resnet_refined.pth"
yamnet_dir = "checkpoints_yamnet_refined"

# Class names (shared across models after alignment)
classes = np.load("classes_efficientnet.npy")  # shape: (200,)

# -----------------------------
# YAMNet Refined Definition
# -----------------------------
class YAMNetRefined(nn.Module):
    def __init__(self, num_classes, hidden_size=512, num_layers=2, dropout=0.3):
        super().__init__()
        self.base_model = hub.load("https://tfhub.dev/google/yamnet/1")
        self.lstm = nn.LSTM(
            input_size=1024,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        embeddings_list = []
        for wav in x:
            wav_np = wav.squeeze().cpu().numpy().astype(np.float32)
            _, embeddings, _ = self.base_model(wav_np)
            embeddings_tensor = torch.tensor(embeddings.numpy(), device=device)
            embeddings_list.append(embeddings_tensor)

        lengths = [e.shape[0] for e in embeddings_list]
        max_len = max(lengths)
        padded = []
        for e in embeddings_list:
            if e.shape[0] < max_len:
                pad = torch.zeros(max_len - e.shape[0], e.shape[1], device=device)
                e = torch.cat([e, pad], dim=0)
            padded.append(e)
        embeddings_batch = torch.stack(padded)  # [batch, time, 1024]

        _, (hn, _) = self.lstm(embeddings_batch)
        hn = torch.cat([hn[-2], hn[-1]], dim=1)  # last bidirectional layer
        return self.fc(hn)

# -----------------------------
# Load Models
# -----------------------------
def load_efficientnet():
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(torch.load(efficientnet_path, map_location=device))
    model.to(device).eval()
    return model

def load_resnet():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(torch.load(resnet_path, map_location=device))
    model.to(device).eval()
    return model

def load_yamnet_folds():
    folds = []
    for i in range(1, 6):
        path = os.path.join(yamnet_dir, f"yamnet_fold{i}.pth")
        if not os.path.exists(path):
            print(f"⚠️ Missing {path}, skipping this fold")
            continue
        model = YAMNetRefined(num_classes=len(classes)).to(device)
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        folds.append(model)
    return folds

# -----------------------------
# Preprocessing
# -----------------------------
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        return img_transform(img).unsqueeze(0).to(device)
    except Exception as e:
        print(f"⚠️ Could not load image {image_path}: {e}")
        return None

def preprocess_audio(audio_path, target_sr=16000, max_len=16000*5):
    try:
        waveform, sr = torchaudio.load(audio_path)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)  # mono
        if waveform.shape[1] < max_len:
            pad = max_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        else:
            waveform = waveform[:, :max_len]
        return waveform.unsqueeze(0).to(device)  # add batch dim
    except Exception as e:
        print(f"⚠️ Could not load audio {audio_path}: {e}")
        return None

# -----------------------------
# Inference Functions
# -----------------------------
def predict_image(model, tensor):
    if tensor is None:
        return None
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs

def predict_audio(models, waveform):
    if waveform is None or not models:
        return None
    all_probs = []
    for m in models:
        with torch.no_grad():
            logits = m(waveform)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            all_probs.append(probs)
    return np.mean(all_probs, axis=0) if all_probs else None

# -----------------------------
# Ensemble Inference
# -----------------------------
def predict(image_path=None, audio_path=None):
    probs_list = []

    # Image models
    if image_path and os.path.exists(image_path):
        img_tensor = preprocess_image(image_path)
        if img_tensor is not None:
            effnet_probs = predict_image(load_efficientnet(), img_tensor)
            resnet_probs = predict_image(load_resnet(), img_tensor)
            if effnet_probs is not None: probs_list.append(effnet_probs)
            if resnet_probs is not None: probs_list.append(resnet_probs)
    elif image_path:
        print(f"⚠️ Image file not found: {image_path}")

    # Audio models
    if audio_path and os.path.exists(audio_path):
        yamnet_folds = load_yamnet_folds()
        waveform = preprocess_audio(audio_path)
        yamnet_probs = predict_audio(yamnet_folds, waveform)
        if yamnet_probs is not None: probs_list.append(yamnet_probs)
    elif audio_path:
        print(f"⚠️ Audio file not found: {audio_path}")

    # Fusion
    if not probs_list:
        raise ValueError("No valid input provided. Please give an image or audio file.")
    fused_probs = np.mean(probs_list, axis=0)

    # Final prediction
    pred_idx = np.argmax(fused_probs)
    pred_name = classes[pred_idx]
    return pred_name

# # -----------------------------
# # Example Usage
# # -----------------------------
# if __name__ == "__main__":
#     print("Prediction (image only):", predict(image_path="bird_examples/example1.jpg"))
#     print("Prediction (audio only):", predict(audio_path="bird_examples/example_bird.wav"))
#     print("Prediction (image + audio):", predict(image_path="bird_examples/example_bird.jpg", audio_path="example_bird.wav"))
