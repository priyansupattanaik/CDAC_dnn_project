import json
from pathlib import Path

import gradio as gr
import librosa
import numpy as np
import torch
import torch.nn as nn

BASE_DIR = Path(__file__).resolve().parent

MODEL_CANDIDATES = [
    BASE_DIR / "models" / "forestguard_cnn.pt",
    BASE_DIR / "forestguard_cnn.pt",
    BASE_DIR / "forestguard_cnn(1).pt",
    BASE_DIR / "forestguard_cnn(2).pt",
]

NORM_CANDIDATES = [
    BASE_DIR / "models" / "norm_stats.npz",
    BASE_DIR / "norm_stats.npz",
    BASE_DIR / "norm_stats(1).npz",
    BASE_DIR / "norm_stats(2).npz",
    BASE_DIR / "norm_stats(3).npz",
]

LABEL_CANDIDATES = [
    BASE_DIR / "models" / "label_map.json",
    BASE_DIR / "label_map.json",
    BASE_DIR / "label_map(1).json",
]


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


MODEL_PATH = first_existing(MODEL_CANDIDATES)
NORM_PATH = first_existing(NORM_CANDIDATES)
LABEL_PATH = first_existing(LABEL_CANDIDATES)

if MODEL_PATH is None:
    raise FileNotFoundError(
        "Could not find the model file. Place forestguard_cnn.pt in models/ or the project root."
    )

if NORM_PATH is None:
    raise FileNotFoundError(
        "Could not find norm_stats.npz. Place it in models/ or the project root."
    )

if LABEL_PATH is None:
    raise FileNotFoundError(
        "Could not find label_map.json. Place it in models/ or the project root."
    )

SR = 22050
N_MELS = 128
FMAX = 8000
TARGET_FRAMES = 216

with open(LABEL_PATH, "r", encoding="utf-8") as f:
    raw_label_map = json.load(f)

label_map = {int(k): v for k, v in raw_label_map.items()}
wild_label = label_map.get(0, "Wildlife")
threat_label = label_map.get(1, "Threat")

stats = np.load(NORM_PATH)
mean_val = float(stats["mean"])
std_val = float(stats["std"])


class ForestGuardCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.35),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 4 * 4, 96),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(96, 2),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ForestGuardCNN().to(device)
state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.eval()


def extract_mel_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=SR)
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        fmax=FMAX,
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    if mel_db.shape[1] < TARGET_FRAMES:
        pad_width = TARGET_FRAMES - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mel_db = mel_db[:, :TARGET_FRAMES]

    return mel_db


def predict(audio_path):
    if audio_path is None:
        return {wild_label: 0.0, threat_label: 0.0}, "Please upload an audio file."

    try:
        mel_db = extract_mel_spectrogram(audio_path)
        mel_db = (mel_db - mean_val) / (std_val + 1e-8)
        x = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        wild_prob = float(probs[0])
        threat_prob = float(probs[1])

        if wild_prob >= threat_prob:
            verdict = f"🌿 Wildlife detected\n\nConfidence: {wild_prob * 100:.2f}%"
        else:
            verdict = f"🚨 Threat detected\n\nConfidence: {threat_prob * 100:.2f}%"

        return {wild_label: wild_prob, threat_label: threat_prob}, verdict

    except Exception as e:
        return {wild_label: 0.0, threat_label: 0.0}, f"Error processing audio: {e}"


theme = gr.themes.Soft(primary_hue="green", secondary_hue="emerald")

css = """
.gradio-container {
    background:
        radial-gradient(circle at top, rgba(34, 197, 94, 0.25), transparent 30%),
        linear-gradient(135deg, #052e16 0%, #0f172a 55%, #111827 100%);
    color: white;
}
#title { text-align: center; font-size: 3rem; font-weight: 800; margin-bottom: 0.5rem; }
#subtitle { text-align: center; font-size: 1.05rem; opacity: 0.92; margin-bottom: 1rem; }
.card {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 20px;
    padding: 18px;
    backdrop-filter: blur(10px);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
}
footer { display: none !important; }
label, .wrap, .prose { color: white !important; }
"""

with gr.Blocks(theme=theme, css=css, title="ForestGuard") as demo:
    gr.Markdown(
        """
<div id="title">🌲 ForestGuard</div>
<div id="subtitle">Wildlife Threat Detection Using Environmental Audio</div>
""",
        elem_classes=["card"],
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown(
                """
### About this demo
Upload a WAV audio clip and the model will classify it as:

- 🌿 Wildlife
- 🚨 Threat

This app uses the same preprocessing as training:
- Sample rate: 22050 Hz
- Mel bins: 128
- Max frequency: 8000 Hz
- Fixed input shape: 128 × 216
""",
                elem_classes=["card"],
            )
        with gr.Column(scale=1):
            audio_input = gr.Audio(type="filepath", label="Upload Forest Audio")
            analyze_btn = gr.Button("Analyze Audio", variant="primary")

    with gr.Row():
        prediction_output = gr.Label(label="Prediction Probabilities", num_top_classes=2)
        verdict_output = gr.Textbox(label="Final Verdict", lines=4)

    analyze_btn.click(
        fn=predict,
        inputs=audio_input,
        outputs=[prediction_output, verdict_output],
    )


if __name__ == "__main__":
    demo.launch()