# MelodAI: Comprehensive System Architecture & Engineering Blueprint

This document provides an exhaustive, low-level technical mapping of **MelodAI**, an advanced polyphonic audio-to-MIDI transcription and arrangement engine. It covers the complete pipeline logic, the backend FastAPI orchestration, the computational musicology heuristics, and the React-based Neo-Brutalist frontend.

---

## 🏗️ 1. Global Architecture Overview

MelodAI follows a **decoupled, event-driven pipeline architecture**. The system is designed to transform raw acoustic waveforms into highly structured, musically coherent symbolic data (MIDI/MusicXML) through a multi-stage refinement process.

### High-Level Flow
1. **Ingress**: User uploads `.mp3`/`.wav` via the React frontend.
2. **Orchestration**: FastAPI (`server.py`) spawns a background thread job, allocating a UUID.
3. **Extraction Block (ML)**: Stems are isolated, pitches extracted, and timbres classified.
4. **Arrangement Block (Heuristics)**: Raw note arrays are processed through a "Expert Arranger" rule-engine.
5. **Synthesis Block**: Refined notes are humanized and encoded into MIDI/MusicXML.
6. **Egress**: Job polling returns the final artifact to the UI for playback via Tone.js.

---

## ⚙️ 2. Detailed Technical Stack

### Backend Orchestration
- **FastAPI / Uvicorn**: Chosen for asynchronous I/O handling and high concurrency.
- **Multithreading**: Uses `threading.Lock` and background daemon threads to manage job states (`jobs` dict) without the overhead of Redis for local development.
- **Subprocess Isolation**: Heavy ML workloads are called as separate Python processes or modularized function calls to prevent main-thread blocking.

### Machine Learning & DSP
- **Source Separation**: Meta's `Demucs` (Hybrid Transformer) isolates 'vocals' and 'other' stems to reduce polyphonic collisions.
- **Transcription**: Spotify's `Basic Pitch` CNN for polyphonic pitch tracking.
- **Timbre Classification**: Custom **ResNet18** model (PyTorch) trained on Mel Spectrograms to detect instrument classes.
- **DSP Utilities**: `librosa` for STFT, beat-tracking, and feature extraction (Spectral Centroid, MFCCs, Chroma).

### Computational Musicology
- **Music21**: Used for deep harmonic analysis, chord parsing, and MusicXML generation.
- **Pretty-MIDI**: Core library for symbolic MIDI manipulation and bit-stream writing.

### Frontend Engineering
- **React 18 + Vite**: Neo-Brutalist design system implemented with Vanilla CSS.
- **Audio Engine**: `Tone.js` for sample-accurate scheduling of MIDI sounds using SoundFonts.
- **State Management**: React Hooks + Polling pattern for background job tracking.

---

## 🌊 3. The MelodAI Pipeline: Step-by-Step

### Phase 1: Stem Separation (`DEMUCS`)
Raw audio is passed through `htdemucs --two-stems vocals`. 
- **Goal**: Isolate the melodic content from percussive transients (drums) and harmonic mud.
- **Result**: A clean-ish instrumental stem for pitch detection.

### Phase 2: Polyphonic Pitch Tracking (`BASIC_PITCH`)
The isolated stem is run through Spotify's `predict` function.
- **Parameters**: `onset_threshold`, `frame_threshold`, `minimum_note_length=0.1`.
- **Optimization**: For monophonic instruments (Flute, Violin), thresholds are tightened to prevent ghost-note artifacts.

### Phase 3: Timbre & Classifier (`CNN_CLASSIFIER`)
The audio is chunked into 3-second windows with 1.5s hop size.
- **Spectral Analysis**: Extracts energy, tempo, chroma mean, and spectral centroid.
- **CNN Inference**: Each window is converted to a log-Mel Spectrogram and run through the ResNet18 model to determine if the detected pitches match the target instrument timbre.

### Phase 4: Global Key Detection (`KEY_DETECT`)
Aggregated chroma vectors are compared against the **Krumhansl-Schmuckler** profile using the correlation coefficient.
- **Goal**: Establish a tonal center (e.g., "C Major") to guide the arrangement engine.

### Phase 5: The Expert Arranger Engine (`ARRANGEMENT`)
This is the "Brain" of MelodAI, implementing 5 key cognitive musicology models:

1.  **Frequency Band Gating**: Octave-shifts notes falling outside the instrument's physical MIDI range (e.g., Sitar: 50-77) into playable bounds.
2.  **Adaptive Quantization**: Uses `librosa.beat.beat_track` to snap notes to a dynamic grid based on local tempo fluctuations rather than a rigid global metronome.
3.  **Huron & Narmour Role Assignment**: 
    - Assigns `melody`, `bass`, and `harmony` roles.
    - **Stepwise Preference**: Penalizes large leaps unless followed by a reversal (Huron).
    - **Registral Return**: Pulls melodic contours back towards the median pitch (Narmour).
4.  **Tymoczko Voice Leading**:
    - Minimizes "Voice leading distance" (semitone displacement) between consecutive chords.
    - **Parallelism Correction**: Actively detects and breaks parallel 5ths and octaves to ensure independent voice motion.
5.  **Lerdahl Tension & Density Budgeting**:
    - Calculates a "Tension Score" based on dissonant intervals (tritones, minor 9ths).
    - **Dynamic Dropout**: If a chord is too dense for the instrument (e.g., Piano limit = 6 notes), it "drops" the least important notes (chromatic noise -> extensions -> 5ths).
    - **Implied Harmony**: If the melody and bass already outline a chord tone, the inner harmony version of that note is purged to save polyphony budget.

### Phase 6: Humanized Rendering (`MIDI_GEN`)
Applies the **Todd (1992)** phrasing model:
- **Ritardando**: Stretches durations of notes at the end of musical phrases using a cubic deceleration curve.
- **Micro-Timing**: 
    - "Pre-climax nudge": Rushes notes approaching a melodic peak by ~6ms.
    - Instrument jitter: Adds ±10ms random variance to simulate human imperfection.
- **Velocity Shaping**: Maps velocities to role-based ranges (Melody is always prioritized as loudest).

---

## 📂 4. Module & File Mapping

| File | Responsibility |
| :--- | :--- |
| `server.py` | FastAPI entry point, job queue, thread management, CORS, polling routes. |
| `api.py` | (Legacy/Slim) Simplified pipeline trigger for container health checks. |
| `algorithmic_transcriber.py` | Standalone CLI for testing extraction logic. |
| `prepare_for_llm.py` | Orchestrates Demucs -> Basic Pitch -> ResNet18 into `llm_input.json`. |
| `expert_arranger.py` | The Massive heuristic engine implementing Huron/Narmour/Lerdahl/Tymoczko. |
| `to_midi.py` | The synthesis engine implementing Todd phrasing and bit-stream writing. |
| `visualize_pipeline.py` | Research tool generating Matplotlib/Seaborn figures for academic validation. |
| `model.py` | Definition of the ResNet18 PyTorch architecture. |
| `dataset.py` | Data loading and label mapping for the instrument classifier. |
| `frontend/` | The Neo-Brutalist React application. |

---

## 📊 5. Data Schema: `arranged.json`
The output of the arranger is a highly structured JSON array used by both `to_midi.py` and the `visualize_pipeline.py`.

```json
{
  "time": 1.25,        // Precise onset in seconds
  "note": "C4",        // Scientific pitch notation
  "midi": 60,          // Calculated MIDI number after voice-leading
  "duration": 0.45,    // Raw duration in seconds
  "velocity": 95,      // Formatted velocity (0-127)
  "hand": "right",     // Score allocation (Piano only)
  "role": "melody",    // Functional role (Melody/Bass/Harmony)
  "phrase_start": true // Metadata for Todd phrasing triggers
}
```

---

## 🔬 6. Research & Visualization Pipeline
`visualize_pipeline.py` is used to validate the algorithms against established musicology. It generates:
- **Figure 8 (Voice Leading Geometry)**: Network graph of chord transitions.
- **Figure 9 (Tension Curve)**: Lerdahl tension score mapped over the song timeline.
- **Figure 10 (Implication-Realization)**: Narmour interval distributions vs. Pearce-Wiggins probabilities.
- **Figure 11 (Todd Phrasing)**: Velocity/IOI curves showing "Kinematic Rubato".

---

## 🐳 7. Infrastructure
The system is fully containerized:
- **Multi-stage Docker**: 
    - Stage 1: Build the Vite frontend.
    - Stage 2: Install massive Python dependencies (PyTorch, Demucs, Music21).
- **Environment**: Optimized for CPU-only inference (standard cloud instances), but leverages CUDA if available.
- **Volumes**: Persistent storage for `/uploads` and `/outputs` to allow the host Windows system to browse generated MIDI artifacts.
