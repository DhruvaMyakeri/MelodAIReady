"""
prepare_for_llm.py
──────────────────
Builds a rich, LLM-ready JSON by combining:
  1. Instrument detection  (your trained model via inference logic)
  2. Real pitch/note data  (Basic Pitch neural model)

The output JSON is designed to be pasted directly into an LLM
(ChatGPT, Claude, Gemini, etc.) along with the transcription prompt.

Usage:
  python prepare_for_llm.py --audio test.mp3 --output llm_input.json

Then paste llm_input.json into your LLM with the transcription prompt.
"""

import torch
import librosa
import numpy as np
import json
import argparse
import pretty_midi
import subprocess
import os
import shutil
from pathlib import Path
from tqdm import tqdm

from model import InstrumentClassifier
from dataset import IDX2LABEL, NUM_CLASSES
from features import extract_features
from basic_pitch.inference import predict
from basic_pitch import ICASSP_2022_MODEL_PATH

# ─── CONFIG ──────────────────────────────────────────────────────────────────

MODEL_PATH  = str(Path(__file__).parent / "checkpoints" / "best_model.pt")
SAMPLE_RATE = 22050
WINDOW_SIZE = 3.0
HOP_SIZE    = 1.5
THRESHOLD   = 0.3

# ─── LOAD INSTRUMENT DETECTOR ────────────────────────────────────────────────

def load_detector(device):
    model = InstrumentClassifier(num_classes=NUM_CLASSES).to(device)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def preprocess_mel(segment, device):
    target_len = int(SAMPLE_RATE * WINDOW_SIZE)
    if len(segment) < target_len:
        segment = np.pad(segment, (0, target_len - len(segment)))
    else:
        segment = segment[:target_len]

    mel = librosa.feature.melspectrogram(
        y=segment, sr=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=128
    )
    mel = librosa.power_to_db(mel)
    mel = (mel - np.mean(mel)) / (np.std(mel) + 1e-8)

    if mel.shape[1] < 128:
        mel = np.pad(mel, ((0, 0), (0, 128 - mel.shape[1])))
    else:
        mel = mel[:, :128]

    return torch.tensor(mel).unsqueeze(0).unsqueeze(0).to(device)


# ─── KRUMHANSL-SCHMUCKLER KEY DETECTION ──────────────────────────────────────

MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def detect_key_ks(global_chroma):
    chroma = np.array(global_chroma)
    if np.sum(chroma) > 0:
        chroma = chroma / np.sum(chroma)
        
    results = []
    
    for i in range(12):
        shifted_maj = np.roll(MAJOR_PROFILE, i)
        shifted_min = np.roll(MINOR_PROFILE, i)
        
        corr_maj = np.corrcoef(chroma, shifted_maj)[0, 1]
        corr_min = np.corrcoef(chroma, shifted_min)[0, 1]
        
        # Convert NaN to -1
        corr_maj = -1 if np.isnan(corr_maj) else corr_maj
        corr_min = -1 if np.isnan(corr_min) else corr_min
        
        results.append((PITCH_CLASSES[i] + " Major", corr_maj))
        results.append((PITCH_CLASSES[i] + " Minor", corr_min))
        
    results.sort(key=lambda x: x[1], reverse=True)
    
    top_candidate, top_score = results[0]
    _, second_score = results[1]
    
    is_ambiguous = True
    if top_score > 0 and top_score >= (second_score * 1.15):
        is_ambiguous = False
        
    best_key = "Ambiguous" if is_ambiguous else top_candidate
    return best_key, round(float(top_score), 4)


# ─── BASIC PITCH: get notes in a time window ─────────────────────────────────

def get_notes_in_window(note_events, start_s, end_s):
    """Filter Basic Pitch note events that fall within [start_s, end_s]."""
    notes_in = []
    for onset, offset, pitch, velocity, _ in note_events:
        if onset >= start_s and onset < end_s:
            note_name = pretty_midi.note_number_to_name(int(pitch))
            notes_in.append({
                "note":      note_name,
                "midi":      int(pitch),
                "onset":     round(float(onset), 3),
                "offset":    round(float(offset), 3),
                "velocity":  round(float(velocity), 3)
            })
    return sorted(notes_in, key=lambda x: x["onset"])


# ─── MAIN ────────────────────────────────────────────────────────────────────

def run_demucs(audio_path, timeout=300, work_dir=None):
    """
    Run demucs --two-stems vocals on the input MP3.
    Returns the path to the no_vocals stem file.
    Raises RuntimeError on failure or timeout.
    """
    print(f"[DEMUCS] Running stem separation on: {audio_path}")
    print(f"[DEMUCS] Model: htdemucs | Mode: --two-stems vocals")
    print(f"[DEMUCS] This may take 1-3 minutes on CPU...")

    cmd = ["python", "-m", "demucs", "--two-stems", "vocals", "-n", "htdemucs", audio_path]

    try:
        result = subprocess.run(
            cmd,
            timeout=timeout,
            capture_output=True,
            text=True,
            cwd=work_dir
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"[DEMUCS] ERROR: Demucs timed out after {timeout}s. "
            "Try --skip-demucs and pass a pre-separated file directly."
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"[DEMUCS] ERROR: Demucs exited with code {result.returncode}.\n"
            f"stderr: {result.stderr.strip()}"
        )

    # Resolve output path: separated/htdemucs/<track_name>/no_vocals.wav
    audio_stem = Path(audio_path).stem
    base_dir = Path(work_dir) if work_dir else Path(".")
    separated_dir = base_dir / "separated" / "htdemucs" / audio_stem

    # Demucs can produce .wav or .mp3 depending on version
    for ext in ["no_vocals.wav", "no_vocals.mp3"]:
        candidate = separated_dir / ext
        if candidate.exists():
            print(f"[DEMUCS] Stem separation complete. Using: {candidate}")
            return str(candidate)

    raise RuntimeError(
        f"[DEMUCS] ERROR: Could not find no_vocals output in {separated_dir}. "
        "Check demucs output structure."
    )


def main():
    parser = argparse.ArgumentParser(description="Build LLM-ready JSON from audio")
    parser.add_argument("--audio",        required=True,            help="Input audio file")
    parser.add_argument("--output",       default="llm_input.json", help="Output JSON path")
    parser.add_argument("--instrument",   default="Piano",          help="Target instrument for heuristics")
    parser.add_argument("--window",       type=float, default=3.0,  help="Segment window size (s)")
    parser.add_argument("--hop",          type=float, default=1.5,  help="Hop size (s)")
    parser.add_argument("--threshold",    type=float, default=0.3,  help="Instrument detection threshold")
    parser.add_argument("--skip-demucs",  action="store_true",      help="Skip stem separation (use input as-is or if already separated)")
    parser.add_argument("--keep-stems",   action="store_true",      help="Keep the demucs separated/ folder after extraction")
    args = parser.parse_args()

    # ── Step 0: Demucs Stem Separation ──
    extraction_audio = args.audio  # default: original file
    demucs_output_dir = None

    if not args.skip_demucs:
        try:
            no_vocals_path = run_demucs(args.audio)
            extraction_audio = no_vocals_path
            # Track the top-level separated/ dir for cleanup
            demucs_output_dir = Path("separated")
        except RuntimeError as e:
            print(f"\n{e}")
            print("[DEMUCS] Falling back to original audio file for extraction.")
            extraction_audio = args.audio
    else:
        print(f"[INFO] --skip-demucs set. Using input file directly: {args.audio}")

    print(f"[INFO] Extraction source: {extraction_audio}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ── Step 1: Basic Pitch (vocal-suppressed source) ──
    print(f"[INFO] Running Basic Pitch on {extraction_audio}...")
    
    # Dynamic Confidence Thresholding based on Instrument Natural Voice
    is_monophonic = args.instrument in ["Sitar", "Flute", "Violin", "Trumpet"]
    p_onset = 0.6 if is_monophonic else 0.35
    p_frame = 0.4 if is_monophonic else 0.2
    
    print(f"[INFO] Applying instrument-aware confidences -> Onset: {p_onset}, Frame: {p_frame}")
    _, _, note_events = predict(
        audio_path=extraction_audio,
        model_or_model_path=ICASSP_2022_MODEL_PATH,
        onset_threshold=p_onset,
        frame_threshold=p_frame,
        minimum_note_length=0.1,
        melodia_trick=True,
    )
    print(f"[INFO] Basic Pitch detected {len(note_events)} note events")

    # ── Step 2: Load audio (vocal-suppressed source for feature extraction) ──
    print(f"[INFO] Loading audio...")
    y, sr = librosa.load(extraction_audio, sr=SAMPLE_RATE)
    duration = len(y) / SAMPLE_RATE
    print(f"[INFO] Duration: {duration:.2f}s")

    # ── Step 3: Load instrument detector ──
    print(f"[INFO] Loading instrument detector...")
    detector = load_detector(device)

    # ── Step 4: Sliding window — merge both sources ──
    segments = []
    step   = int(args.hop * SAMPLE_RATE)
    window = int(args.window * SAMPLE_RATE)

    for start_i in tqdm(range(0, len(y) - window, step), desc="Analyzing segments"):
        end_i     = start_i + window
        start_s   = round(start_i / SAMPLE_RATE, 2)
        end_s     = round(end_i / SAMPLE_RATE, 2)
        segment   = y[start_i:end_i]

        # ── Instrument detection ──
        mel = preprocess_mel(segment, device)
        with torch.no_grad():
            probs = torch.sigmoid(detector(mel)).cpu().numpy()[0]

        instruments = [IDX2LABEL[i] for i, p in enumerate(probs) if p >= args.threshold]
        instrument_confidences = {
            IDX2LABEL[i]: round(float(p), 3)
            for i, p in enumerate(probs)
            if p >= args.threshold
        }

        # ── Audio features ──
        feats = extract_features(segment, SAMPLE_RATE)

        # ── Notes from Basic Pitch in this window ──
        notes_in_window = get_notes_in_window(note_events, start_s, end_s)

        # ── Dominant note summary for this window ──
        dominant_note = None
        if notes_in_window:
            # Pick the note with highest velocity in the window
            dominant = max(notes_in_window, key=lambda n: n["velocity"])
            dominant_note = dominant["note"]

        segments.append({
            "time_start":   start_s,
            "time_end":     end_s,
            # Instrument info
            "instruments":  instruments,
            "instrument_confidence": instrument_confidences,
            # Real pitch data (from Basic Pitch)
            "notes":        notes_in_window,          # all notes in this window
            "dominant_note": dominant_note,           # most prominent note
            # Audio features
            "energy":       round(feats["energy"], 4),
            "tempo":        round(feats["tempo"], 2),
            "chroma_mean":  [round(v, 4) for v in feats["chroma_mean"]],
            "spectral_centroid": round(feats["spectral_centroid"], 2),
        })

    # ── Step 5: Save ──
    global_chroma = np.mean([s["chroma_mean"] for s in segments], axis=0).tolist()
    detected_key, key_conf = detect_key_ks(global_chroma)
    print(f"[INFO] Detected KS Global Key: {detected_key} (Confidence: {key_conf})")
    
    output = {
        "audio_file":   args.audio,
        "duration":     round(duration, 2),
        "num_segments": len(segments),
        "note_hint":    "chroma_mean order: [C, C#, D, D#, E, F, F#, G, G#, A, A#, B]",
        "detected_key": detected_key,
        "detected_key_confidence": key_conf,
        "segments":     segments
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    total_notes = sum(len(s["notes"]) for s in segments)
    print(f"\n[DONE] Saved -> {args.output}")
    print(f"       Segments : {len(segments)}")
    print(f"       Total notes detected: {total_notes}")

    # ── Cleanup demucs stems unless --keep-stems ──
    if demucs_output_dir and demucs_output_dir.exists() and not args.keep_stems:
        shutil.rmtree(demucs_output_dir)
        print(f"[DEMUCS] Cleaned up stems directory: {demucs_output_dir}")
    elif demucs_output_dir and args.keep_stems:
        print(f"[DEMUCS] --keep-stems set. Stems preserved at: {demucs_output_dir}")


if __name__ == "__main__":
    main()
