# MelodAI: Audio-to-Symbolic Music Transcription Pipeline

A cutting-edge, end-to-end machine learning system that transforms complex, multi-instrument audio recordings (MP3/WAV) into mathematically structured, arranged MIDI scores. Now fully dockerized with an asynchronous FastAPI backend and a custom Neo-Brutalist React web interface featuring high-fidelity `Tone.js` SoundFont previews.

---

## Overview

MelodAI goes far beyond basic instrument classification. It extracts, isolates, tracks, and synthetically reconstructs actual musical compositions from polyphonic audio tracking.

**The Pipeline Flow**:
Raw Audio → Stem Isolation → Note Event Detection → CNN Instrument Analysis → Symbolic Music Orchestration → MIDI Generation 

---

## Pipeline Architecture

1. **Stem Separation (Demucs)**: 
   Surgically separates the input audio into distinct mixing stems (Vocals, Drums, Bass, Other/Melody) to reduce polyphonic collisions before pitch tracking.

2. **Note Tracking (Basic Pitch)**:
   A highly tuned Spotify neural network model scans the separated stems to extract fundamental frequencies, creating raw note sequences mapped over time.

3. **Timbre Analysis (ResNet CNN)**:
   A custom PyTorch ResNet18 model analyzes Mel Spectrogram windows to classify the dominant instrument in the recording, accurately assigning the correct General MIDI patches.

4. **Expert Arranger Engine**:
   A deterministic, algorithmic rule-engine applies advanced music theory heuristics (voice-leading adjustments, density budgeting, melodic interval shaping, velocity normalization) to mathematically refine the raw transcription into sheet-music-ready structures without risking LLM quota limits or hallucinations.

5. **Musical Generation**:
   Converts the final refined JSON structures into cleanly formatted `.mid` outputs using `pretty_midi`, injecting BPM and track assignments seamlessly.

---

## The Interface

MelodAI ships with a beautifully designed, **Neo-Brutalist React Web App**:
- **Async Execution**: Long-running ML operations are submitted via a robust job queue over an async FastAPI orchestration architecture.
- **Real-Time Polling**: The React frontend visually charts the ML pipeline progression across all stages (Demucs → Pitch → CNN → Arranging) using custom loading blocks.
- **High-Fidelity Preview**: Custom `Tone.js` Sampler integrations dynamically fetch official General MIDI multi-gigabyte SoundFonts on the fly. Select 'Guitar', hear a Nylon Acoustic Guitar mapped directly to the generated MIDI notes overlapping your original song.
- **Mixer**: Dual volume staging sliders allow granular mixing of the raw MP3 upload against the AI-generated MIDI for side-by-side comparison.

---

## Tech Stack

**Backend (ML Pipeline):**
- **Python 3.10**: Core pipeline glue.
- **FastAPI / Uvicorn**: Asynchronous job handling and queue polling.
- **PyTorch & Librosa**: Audio processing and neural network inference.
- **Demucs & Basic Pitch**: Audio AI modeling.
- **Pretty_MIDI**: Symbolic music formatting.

**Frontend:**
- **React 18 & Vite**: Lightning-fast web rendering.
- **Vanilla CSS**: Fully bespoke, frameworkless Neo-Brutalist user interface.
- **Tone.js & @tonejs/midi**: Accurate MIDI transport scheduling and browser audio synthesis.

**Infrastructure:**
- **Docker & Docker Compose**: Total system containerization mapping GPU/CPU environments synchronously with Nginx statically mapped UI routing.

---

## Quick Start

The entire stack is seamlessly orchestrated via a local Docker network.

1. Ensure Docker Desktop is installed.
2. Clone the repository.
3. Boot the environment from the root directory:

```bash
docker-compose up --build
```

**Port Mapping**:
- The Neo-Brutalist Web App boots instantly to `http://localhost:3000`
- The backend API endpoints natively map to `http://localhost:8000`

### To Use:
Navigate to the web interface, upload a song, and select an output instrument! 

---

## Author

**Dhruva M**
