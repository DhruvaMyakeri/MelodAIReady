import torch
import librosa
import numpy as np
import json
from tqdm import tqdm

from model import InstrumentClassifier
from dataset import IDX2LABEL, NUM_CLASSES
from features import extract_features

import argparse

# ─── PARSE ARGS ─────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--audio", default="test2.mp3", help="Input audio file")
parser.add_argument("--output", default="output2.json", help="Output JSON file")
parser.add_argument("--model", default="./checkpoints/best_model.pt", help="Path to model checkpoint")
args = parser.parse_args()

# ─── CONFIG ─────────────────────────────────────────────

MODEL_PATH = args.model
AUDIO_PATH = args.audio
OUTPUT_PATH = args.output
SAMPLE_RATE = 22050
WINDOW_SIZE = 3.0
HOP_SIZE = 1.5
THRESHOLD = 0.3

# ─── LOAD MODEL ─────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = InstrumentClassifier(num_classes=NUM_CLASSES).to(device)
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("[INFO] Model loaded")

# ─── PREPROCESS (same as training) ──────────────────────

def preprocess(y, sr):
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)

    target_len = int(SAMPLE_RATE * WINDOW_SIZE)

    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=128
    )

    mel = librosa.power_to_db(mel)

    mel = (mel - np.mean(mel)) / (np.std(mel) + 1e-8)

    if mel.shape[1] < 128:
        mel = np.pad(mel, ((0, 0), (0, 128 - mel.shape[1])))
    else:
        mel = mel[:, :128]

    mel = torch.tensor(mel).unsqueeze(0).unsqueeze(0)  # (1,1,128,128)

    return mel

# ─── LOAD AUDIO ─────────────────────────────────────────

print("[INFO] Loading audio...")
y, sr = librosa.load(AUDIO_PATH, sr=SAMPLE_RATE)

duration = len(y) / SAMPLE_RATE

print(f"[INFO] Duration: {duration:.2f}s")

# ─── SLIDING WINDOW ─────────────────────────────────────

results = []

step = int(HOP_SIZE * SAMPLE_RATE)
window = int(WINDOW_SIZE * SAMPLE_RATE)

for start in tqdm(range(0, len(y) - window, step)):

    end = start + window
    segment = y[start:end]

    # ── MODEL ──
    mel = preprocess(segment, SAMPLE_RATE).to(device)

    with torch.no_grad():
        logits = model(mel)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    instruments = [
        IDX2LABEL[i]
        for i, p in enumerate(probs)
        if p >= THRESHOLD
    ]

    # ── FEATURES ──
    features = extract_features(segment, SAMPLE_RATE)

    # ── STORE ──
    results.append({
        "time_start": round(start / SAMPLE_RATE, 2),
        "time_end": round(end / SAMPLE_RATE, 2),
        "instruments": instruments,
        "num_instruments": len(instruments),
        "energy": round(features["energy"], 4),
        "tempo": round(features["tempo"], 2),
        "detected_note": features.get("detected_note"),
        "dominant_frequency": round(features["dominant_frequency"], 2) if features.get("dominant_frequency") else None,
        "chroma_mean": [round(v, 4) for v in features["chroma_mean"]] if features.get("chroma_mean") else None
    })

# ─── FINAL OUTPUT ───────────────────────────────────────

output = {
    "audio_file": AUDIO_PATH,
    "duration": duration,
    "num_segments": len(results),
    "segments": results
}

# Save JSON
with open(OUTPUT_PATH, "w") as f:
    json.dump(output, f, indent=2)

print(f"[DONE] Saved to {OUTPUT_PATH}")