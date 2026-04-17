"""
server.py — FastAPI backend for the Audio-Instruments-ML pipeline.
Launch with: uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import uuid
import json
import shutil
import tempfile
import threading
import traceback
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# ─── APP INIT ────────────────────────────────────────────────────────────────

app = FastAPI(title="Audio-Instruments-ML API", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)



# ─── JOB STATE ───────────────────────────────────────────────────────────────

jobs: Dict[str, Dict[str, Any]] = {}
jobs_lock = threading.Lock()

PHASES = ["DEMUCS", "BASIC_PITCH", "LIBROSA", "CNN_CLASSIFIER", "KEY_DETECT", "ARRANGEMENT", "MIDI_GEN", "OUTPUT"]

def make_job(job_id: str, output_mode: str) -> Dict:
    return {
        "job_id": job_id,
        "phase": "DEMUCS",
        "progress": 0,
        "log": [],
        "error": None,
        "output_mode": output_mode,
        "audio_path": None,
        "output_path": None,
        "instrument": "Piano",
    }

def update_job(job_id: str, **kwargs):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(kwargs)

def append_log(job_id: str, line: str):
    with jobs_lock:
        if job_id in jobs:
            log = jobs[job_id]["log"]
            log.append(line)
            if len(log) > 50:
                jobs[job_id]["log"] = log[-50:]

# ─── PIPELINE RUNNER ─────────────────────────────────────────────────────────

def run_pipeline(job_id: str, audio_path: str, instrument: str, output_mode: str,
                 skip_demucs: bool, threshold: float, work_dir: str):
    """Run the full pipeline in a background thread."""
    try:
        # ── Import pipeline modules ──
        sys.path.insert(0, str(Path(__file__).parent))
        import torch
        import librosa
        import numpy as np
        import pretty_midi

        # ─── PHASE: DEMUCS ───────────────────────────────────────────────────
        update_job(job_id, phase="DEMUCS", progress=2)
        append_log(job_id, "[DEMUCS] Starting stem separation...")

        extraction_audio = audio_path

        if not skip_demucs:
            import subprocess
            from prepare_for_llm import run_demucs
            update_job(job_id, progress=5)
            append_log(job_id, "[DEMUCS] Running htdemucs --two-stems vocals...")
            try:
                extraction_audio = run_demucs(audio_path, work_dir=work_dir)
                append_log(job_id, f"[DEMUCS] Separation complete: {extraction_audio}")
            except RuntimeError as e:
                append_log(job_id, f"[DEMUCS] Failed, using original: {e}")
                extraction_audio = audio_path
        else:
            append_log(job_id, "[DEMUCS] Skipped. Using original audio.")

        update_job(job_id, phase="DEMUCS", progress=12)
        append_log(job_id, "[DEMUCS] Done.")

        # ─── PHASE: BASIC PITCH ───────────────────────────────────────────────
        update_job(job_id, phase="BASIC_PITCH", progress=14)
        append_log(job_id, "[BASIC_PITCH] Running Basic Pitch pitch detection...")

        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH
        from prepare_for_llm import (
            load_detector, preprocess_mel, get_notes_in_window, detect_key_ks
        )
        from dataset import IDX2LABEL, NUM_CLASSES
        from features import extract_features

        is_monophonic = instrument in ["Sitar", "Flute", "Violin", "Trumpet"]
        p_onset = 0.6 if is_monophonic else 0.35
        p_frame  = 0.4 if is_monophonic else 0.2

        append_log(job_id, f"[BASIC_PITCH] Thresholds → onset={p_onset}, frame={p_frame}")
        _, _, note_events = predict(
            audio_path=extraction_audio,
            model_or_model_path=ICASSP_2022_MODEL_PATH,
            onset_threshold=p_onset,
            frame_threshold=p_frame,
            minimum_note_length=0.1,
            melodia_trick=True,
        )
        append_log(job_id, f"[BASIC_PITCH] Detected {len(note_events)} note events.")
        update_job(job_id, phase="BASIC_PITCH", progress=28)

        # ─── PHASE: LIBROSA ──────────────────────────────────────────────────
        update_job(job_id, phase="LIBROSA", progress=30)
        append_log(job_id, "[LIBROSA] Loading audio and extracting features...")

        SAMPLE_RATE = 22050
        WINDOW_SIZE = 3.0
        HOP_SIZE    = 1.5

        y, sr = librosa.load(extraction_audio, sr=SAMPLE_RATE)
        duration = len(y) / SAMPLE_RATE
        append_log(job_id, f"[LIBROSA] Duration: {duration:.2f}s")
        update_job(job_id, phase="LIBROSA", progress=36)

        # ─── PHASE: CNN_CLASSIFIER ───────────────────────────────────────────
        update_job(job_id, phase="CNN_CLASSIFIER", progress=38)
        append_log(job_id, "[CNN] Loading instrument classifier model...")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        append_log(job_id, f"[CNN] Device: {device}")
        detector = load_detector(device)

        segments = []
        step   = int(HOP_SIZE    * SAMPLE_RATE)
        window = int(WINDOW_SIZE * SAMPLE_RATE)
        total_steps = len(range(0, len(y) - window, step))

        append_log(job_id, f"[CNN] Classifying {total_steps} segments...")
        for idx, start_i in enumerate(range(0, len(y) - window, step)):
            end_i   = start_i + window
            start_s = round(start_i / SAMPLE_RATE, 2)
            end_s   = round(end_i   / SAMPLE_RATE, 2)
            segment = y[start_i:end_i]

            mel = preprocess_mel(segment, device)
            with torch.no_grad():
                probs = torch.sigmoid(detector(mel)).cpu().numpy()[0]

            instruments_det = [IDX2LABEL[i] for i, p in enumerate(probs) if p >= threshold]
            feats = extract_features(segment, SAMPLE_RATE)
            notes_in_window = get_notes_in_window(note_events, start_s, end_s)
            dominant_note = None
            if notes_in_window:
                dominant = max(notes_in_window, key=lambda n: n["velocity"])
                dominant_note = dominant["note"]

            segments.append({
                "time_start": start_s, "time_end": end_s,
                "instruments": instruments_det,
                "instrument_confidence": {IDX2LABEL[i]: round(float(p), 3) for i, p in enumerate(probs) if p >= threshold},
                "notes": notes_in_window, "dominant_note": dominant_note,
                "energy": round(feats["energy"], 4), "tempo": round(feats["tempo"], 2),
                "chroma_mean": [round(v, 4) for v in feats["chroma_mean"]],
                "spectral_centroid": round(feats["spectral_centroid"], 2),
            })

            # Progress: CNN occupies 38→58
            cnn_prog = 38 + int((idx / max(total_steps, 1)) * 20)
            update_job(job_id, progress=cnn_prog)

        append_log(job_id, f"[CNN] Classified {len(segments)} segments.")
        update_job(job_id, phase="CNN_CLASSIFIER", progress=58)

        # ─── PHASE: KEY_DETECT ───────────────────────────────────────────────
        update_job(job_id, phase="KEY_DETECT", progress=60)
        append_log(job_id, "[KEY] Detecting global key (Krumhansl-Schmuckler)...")

        global_chroma = np.mean([s["chroma_mean"] for s in segments], axis=0).tolist()
        detected_key, key_conf = detect_key_ks(global_chroma)
        append_log(job_id, f"[KEY] Detected: {detected_key} (conf={key_conf})")

        llm_json_path = os.path.join(work_dir, "llm_input.json")
        llm_output = {
            "audio_file": audio_path, "duration": round(duration, 2),
            "num_segments": len(segments),
            "note_hint": "chroma_mean order: [C, C#, D, D#, E, F, F#, G, G#, A, A#, B]",
            "detected_key": detected_key, "detected_key_confidence": key_conf,
            "segments": segments,
        }
        with open(llm_json_path, "w") as f:
            json.dump(llm_output, f, indent=2)

        update_job(job_id, phase="KEY_DETECT", progress=65)
        append_log(job_id, f"[KEY] Saved extraction JSON: {llm_json_path}")

        # ─── PHASE: ARRANGEMENT ──────────────────────────────────────────────
        update_job(job_id, phase="ARRANGEMENT", progress=67)
        append_log(job_id, "[ARRANGE] Running Expert Arranger (music theory pipeline)...")

        from expert_arranger import arrange
        arranged_json_path = os.path.join(work_dir, "arranged.json")

        arrange(
            json_file=llm_json_path,
            audio_file=extraction_audio,
            output_file=arranged_json_path,
            instrument=instrument,
        )
        append_log(job_id, f"[ARRANGE] Complete. Saved: {arranged_json_path}")
        update_job(job_id, phase="ARRANGEMENT", progress=80)

        # ─── PHASE: MIDI_GEN / OUTPUT ─────────────────────────────────────────
        if output_mode == "midi":
            update_job(job_id, phase="MIDI_GEN", progress=82)
            append_log(job_id, "[MIDI] Generating MIDI file...")

            midi_out_path = os.path.join(work_dir, "output.mid")
            # Import and run to_midi logic
            import json as _json
            import random
            from to_midi import apply_todd_phrasing, note_name_to_midi, INSTRUMENT_PROGRAMS

            with open(arranged_json_path, "r") as f:
                notes = _json.load(f)

            notes.sort(key=lambda x: x["time"])
            tempo_val = 120.0
            notes = apply_todd_phrasing(notes, tempo_val)

            midi = pretty_midi.PrettyMIDI(initial_tempo=tempo_val)
            program = INSTRUMENT_PROGRAMS.get(instrument, 0)
            instr_track = pretty_midi.Instrument(program=program, name=instrument)

            skipped = 0
            for entry in notes:
                midi_num = entry.get("midi")
                if midi_num is None:
                    midi_num = note_name_to_midi(entry.get("note"))
                if midi_num is None:
                    skipped += 1; continue
                midi_num = int(midi_num)
                start    = float(entry["time"])
                dur      = float(entry["duration"])
                role     = entry.get("role", "harmony")

                rit_mult = entry.get("_rit_mult", 1.0)
                dur *= rit_mult
                if entry.get("_pre_climax_nudge", False):
                    start -= 0.006
                offset_ms = random.uniform(-10, 10) if instrument == "Piano" else 0
                start += offset_ms / 1000.0

                decay = {"melody": 0.6, "bass": 0.4}.get(role, 0.3)
                if entry.get("_phrase_end_decay", False):
                    decay += 0.3
                end = start + dur + decay

                raw_vel = float(entry.get("velocity", 80))
                if raw_vel <= 1.0: raw_vel *= 127
                norm_v = min(1.0, raw_vel / 127.0)
                if role == "melody":   base_vel = int(90 + norm_v * 20)
                elif role == "bass":   base_vel = int(60 + norm_v * 20)
                else:                  base_vel = int(50 + norm_v * 20)
                vel = base_vel + random.randint(-10, 10)

                sec_per_beat = 60.0 / tempo_val
                beat_idx = round(start / sec_per_beat)
                is_on_grid = abs(start - beat_idx * sec_per_beat) < 0.05
                if is_on_grid:
                    if beat_idx % 4 == 0:   vel += 8
                    elif beat_idx % 4 == 2: vel += 3 if tempo_val > 120 else 8
                else:
                    vel -= 5
                if entry.get("phrase_start", False):
                    vel += 5
                vel = max(1, min(127, vel))

                if end <= start:
                    end = start + 0.1

                note_obj = pretty_midi.Note(velocity=vel, pitch=midi_num, start=max(0, start), end=end)
                instr_track.notes.append(note_obj)

            midi.instruments.append(instr_track)
            midi.write(midi_out_path)
            append_log(job_id, f"[MIDI] Written {len(instr_track.notes)} notes to {midi_out_path}")

            update_job(job_id, phase="OUTPUT", progress=95,
                       output_path=midi_out_path)
            append_log(job_id, "[OUTPUT] MIDI file ready for download.")

        else:  # sheet
            update_job(job_id, phase="MIDI_GEN", progress=82)
            append_log(job_id, "[SHEET] Generating sheet music (MusicXML)...")

            sheet_out_path = os.path.join(work_dir, "output_sheet.xml")

            import json as _json
            from music21 import stream, note as m21note, metadata as m21meta, clef as m21clef
            import music21

            with open(arranged_json_path, "r") as f:
                notes_data = _json.load(f)

            score = stream.Score()
            score.metadata = m21meta.Metadata()
            score.metadata.title = f"Auto-transcription ({instrument})"
            score.metadata.composer = "Audio-Instruments-ML Pipeline"

            right_part = stream.PartStaff()
            right_part.id = "right"
            right_part.insert(0, m21clef.TrebleClef())
            left_part = stream.PartStaff()
            left_part.id = "left"
            left_part.insert(0, m21clef.BassClef())

            GRID = 0.125
            skipped = 0
            for i, entry in enumerate(notes_data):
                note_name  = entry.get("note")
                duration   = float(entry.get("duration", 0.5))
                start_time = float(entry.get("time", 0.0))
                hand       = entry.get("hand", "right")
                if not note_name:
                    skipped += 1; continue
                try:
                    raw_offset   = start_time * 2.0
                    quarter_off  = round(raw_offset / GRID) * GRID
                    raw_len      = duration * 2.0
                    quarter_len  = max(GRID, min(round(raw_len / GRID) * GRID, 4.0))
                    n = m21note.Note(note_name)
                    n.quarterLength = quarter_len
                    if hand == "left":
                        left_part.insert(quarter_off, n)
                    else:
                        right_part.insert(quarter_off, n)
                except Exception:
                    skipped += 1

                if i % 200 == 0:
                    prog = 82 + int((i / max(len(notes_data), 1)) * 12)
                    update_job(job_id, progress=min(prog, 93))

            score.insert(0, right_part)
            score.insert(0, left_part)
            score.write("musicxml", fp=sheet_out_path)

            append_log(job_id, f"[SHEET] Written {len(notes_data) - skipped} notes to XML.")
            update_job(job_id, phase="OUTPUT", progress=96, output_path=sheet_out_path)
            append_log(job_id, "[OUTPUT] Sheet music ready for download.")

        # ─── DONE ────────────────────────────────────────────────────────────
        update_job(job_id, phase="DONE", progress=100)
        append_log(job_id, "[DONE] Pipeline complete.")

    except Exception as exc:
        tb = traceback.format_exc()
        update_job(job_id, phase="ERROR", error=str(exc))
        append_log(job_id, f"[ERROR] {exc}")
        append_log(job_id, tb[-500:])


# ─── ROUTES ──────────────────────────────────────────────────────────────────

@app.post("/run")
async def run_endpoint(
    file: UploadFile = File(...),
    instrument: str  = Form("Piano"),
    output_mode: str = Form("midi"),
    skip_demucs: bool = Form(False),
    threshold: float  = Form(0.3),
):
    job_id  = str(uuid.uuid4())
    work_dir = tempfile.mkdtemp(prefix=f"aml_{job_id[:8]}_")

    # Save uploaded file
    audio_path = os.path.join(work_dir, file.filename or "upload.mp3")
    with open(audio_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    with jobs_lock:
        jobs[job_id] = make_job(job_id, output_mode)
        jobs[job_id]["audio_path"] = audio_path
        jobs[job_id]["instrument"] = instrument
        jobs[job_id]["work_dir"]   = work_dir

    thread = threading.Thread(
        target=run_pipeline,
        args=(job_id, audio_path, instrument, output_mode, skip_demucs, threshold, work_dir),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id}


@app.get("/status/{job_id}")
async def status_endpoint(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id":   job["job_id"],
        "phase":    job["phase"],
        "progress": job["progress"],
        "log":      job["log"][-10:],
        "error":    job["error"],
    }


@app.get("/download/{job_id}")
async def download_endpoint(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["phase"] not in ("DONE",):
        raise HTTPException(status_code=400, detail="Job not complete")

    out_path = job.get("output_path")
    if not out_path or not os.path.exists(out_path):
        raise HTTPException(status_code=404, detail="Output file not found")

    if job["output_mode"] == "midi":
        return FileResponse(out_path, media_type="application/octet-stream",
                            filename="output.mid")
    else:
        return FileResponse(out_path, media_type="application/xml",
                            filename="output_sheet.xml")


@app.get("/audio/{job_id}")
async def audio_endpoint(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    audio_path = job.get("audio_path")
    if not audio_path or not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(audio_path, media_type="audio/mpeg")


@app.get("/")
async def root():
    return {"message": "Audio-Instruments-ML API v3. POST /run to start a job."}
