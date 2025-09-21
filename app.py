import os
from flask import Flask, request, render_template_string, send_from_directory
import torch
import torch.nn as nn
import torchaudio
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
from PIL import Image
import tensorflow_hub as hub  # ✅ needed for YAMNet embeddings

# -----------------------------
# Config
# -----------------------------
app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

efficientnet_path = "efficientnet_birds.pth"
resnet_path = "resnet_refined.pth"
yamnet_dir = "checkpoints_yamnet_refined"
classes = np.load("classes_efficientnet.npy")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -----------------------------
# YAMNet Refined
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
        hn = torch.cat([hn[-2], hn[-1]], dim=1)
        return self.fc(hn)

# -----------------------------
# Model Loaders
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
    img = Image.open(image_path).convert("RGB")
    return img_transform(img).unsqueeze(0).to(device)

def preprocess_audio(audio_path, target_sr=16000, max_len=16000*5):
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
    return waveform.unsqueeze(0).to(device)

# -----------------------------
# Prediction Functions
# -----------------------------
def predict_image(model, tensor):
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs

def predict_audio(models, waveform):
    all_probs = []
    for m in models:
        with torch.no_grad():
            logits = m(waveform)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            all_probs.append(probs)
    return np.mean(all_probs, axis=0) if all_probs else None

def predict(image_path=None, audio_path=None):
    probs_list = []

    if image_path:
        img_tensor = preprocess_image(image_path)
        effnet_probs = predict_image(load_efficientnet(), img_tensor)
        resnet_probs = predict_image(load_resnet(), img_tensor)
        probs_list.extend([effnet_probs, resnet_probs])

    if audio_path:
        yamnet_folds = load_yamnet_folds()
        waveform = preprocess_audio(audio_path)
        yamnet_probs = predict_audio(yamnet_folds, waveform)
        if yamnet_probs is not None:
            probs_list.append(yamnet_probs)

    if not probs_list:
        return "⚠️ Please upload an image or audio file.", None

    fused_probs = np.mean(probs_list, axis=0)
    pred_idx = np.argmax(fused_probs)
    return classes[pred_idx], image_path

# -----------------------------
# HTML Template
# -----------------------------
HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Bird Species Identifier</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body { background: #f5f7fa; height: 100vh; display: flex; flex-direction: column; }
    .top, .bottom { flex: 1; padding: 20px; }
    .card { padding: 20px; border-radius: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
    h2 { text-align: center; margin-bottom: 20px; }
    .btn { width: 100%; }
    img { max-width: 100%; border-radius: 10px; margin-top: 15px; }
    .desc { text-align: left; margin-top: 15px; font-style: italic; }
  </style>
</head>
<body>
  <div class="top">
    <div class="card">
      <h2>🐦 Bird Species Identifier</h2>
      <form method="post" enctype="multipart/form-data">
        <div class="mb-3">
          <label class="form-label">Upload Bird Image</label>
          <input class="form-control" type="file" name="image" accept="image/*">
        </div>
        <div class="mb-3">
          <label class="form-label">Upload Bird Audio (.wav)</label>
          <input class="form-control" type="file" name="audio" accept="audio/*">
        </div>
        <button class="btn btn-primary" type="submit">Predict</button>
      </form>
    </div>
  </div>
  <div class="bottom">
    {% if result %}
      <div class="card">
        <h4 class="text-center">Prediction: <b>{{ result }}</b></h4>
        <div class="row mt-3">
          {% if image_url %}
          <div class="col-md-6 text-center">
            <img src="{{ image_url }}" alt="Uploaded Bird Image">
          </div>
          {% endif %}
          <div class="col-md-6">
            <p class="desc">{{ description }}</p>
          </div>
        </div>
      </div>
    {% endif %}
  </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def upload_file():
    result, image_url = None, None
    if request.method == "POST":
        image = request.files.get("image")
        audio = request.files.get("audio")

        img_path, audio_path = None, None

        if image and image.filename != "":
            img_path = os.path.join(UPLOAD_FOLDER, image.filename)
            image.save(img_path)
            image_url = f"/uploads/{image.filename}"

        if audio and audio.filename != "":
            audio_path = os.path.join(UPLOAD_FOLDER, audio.filename)
            audio.save(audio_path)

        result, _ = predict(img_path, audio_path)

    return render_template_string(HTML, result=result, image_url=image_url)

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    print("🚀 Starting Bird Identifier Flask App...")
    app.run(debug=True)
