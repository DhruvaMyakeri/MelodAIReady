# 🎧 Audio Instrument Intelligence System

A deep learning system that analyzes audio files to detect instruments and extract meaningful audio features over time.

---

## 🚀 Overview

This project converts raw audio into a structured, time-based representation:

Audio → Segments → Instrument Detection + Feature Extraction → Timeline

Unlike basic classifiers, this system works on **real songs with multiple overlapping instruments**.

---

## 🔥 Key Features

- 🎵 Multi-instrument detection using deep learning (ResNet18)
- 🧠 Trained with **audio mixing** to simulate real-world music
- 📊 Sliding window segmentation (temporal analysis)
- 🎧 Audio feature extraction:
  - MFCC (timbre)
  - Tempo (BPM)
  - Energy (loudness)
  - Spectral centroid (brightness)
  - Chroma (pitch distribution)
- 📈 Visualization:
  - Instrument heatmap
  - Timeline analysis

---

## 🧠 Model Architecture

- Backbone: ResNet18 (modified for 1-channel input)
- Input: Mel Spectrogram (128 × 128)
- Output: Multi-label classification (28 instruments)

---

## 🗂️ Project Structure

Audio-Instruments-ML/
│
├── dataset.py # Data loading + augmentation + mixing
├── model.py # CNN model (ResNet18)
├── train.py # Training pipeline
├── inference.py # Song analysis pipeline
├── features.py # Audio feature extraction
├── visualize.py # Visualization (heatmaps)
├── requirements.txt
├── README.md

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
🏋️ Training
python train.py
Training Techniques Used
🔁 Audio mixing (multi-instrument simulation)
🎚️ Label smoothing
🔊 Gain + noise augmentation
⚖️ Class-balanced sampling
🎧 Inference (Analyze a Song)
python inference.py

Modify inside file:

AUDIO_PATH = "test.mp3"
📊 Output Format
[
  {
    "time": 0.0,
    "instruments": ["Acoustic_Guitar"]
  }
]
📈 Visualization
python visualize.py

Outputs:

Instrument timeline heatmap
Activity over time
🧠 Important Insight

This model performs timbre-based classification, not exact instrument detection.

That means:

Real Sound	Model Output
Harmonium	Accordion
Bright strings	Ukulele
Guitar + effects	Acoustic_Guitar

It predicts the closest known sound profile.

📊 Results
Validation F1 Score: ~0.94
Stable temporal predictions
Improved performance on real songs via data mixing
⚠️ Limitations
Trained on isolated instrument dataset
Confusion between similar instruments (e.g., guitar vs ukulele)
Does not perform full music transcription
🔮 Future Improvements
Train on real multi-instrument datasets
Improve instrument separation
Add pitch & chord detection
Build full audio understanding pipeline
💡 Why This Project Matters

Most ML projects stop at classification.

This project builds:

Audio → Structured Intelligence → Temporal Understanding

Applications:

Music recommendation systems
Audio search engines
AI music analysis tools
👨‍💻 Author

Dhruva M
```
