"""
api.py
──────
FastAPI backend for the Audio-to-MIDI ML pipeline.

Endpoints:
  POST /process-audio   — accepts an audio file + instrument name,
                          runs the full pipeline, returns a .mid file.

Pipeline:
  1. prepare_for_llm.py  →  llm_input_<uuid>.json
  2. expert_arranger.py  →  arranged_<uuid>.json
  3. to_midi.py          →  output_<uuid>.mid

Note: play_audio.py is intentionally NOT used here (requires display/audio hw).
"""

import os
import uuid
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Audio-to-MIDI API",
    description="Upload an audio file and receive a MIDI transcription via the ML pipeline.",
    version="1.0.0",
)

# ── Runtime dirs ─────────────────────────────────────────────────────────────

UPLOADS_DIR = Path("uploads")
OUTPUTS_DIR = Path("outputs")
UPLOADS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def run_step(cmd: list[str], step_name: str) -> None:
    """Run a subprocess pipeline step; raise HTTPException on failure."""
    print(f"[API] >>> Running step: {step_name}")
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(f"[API] !!! Step failed: {step_name}\n{result.stderr}", file=sys.stderr)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline step '{step_name}' failed: {result.stderr.strip()}",
        )
    print(f"[API] <<< Completed step: {step_name}")


# ── Endpoint ──────────────────────────────────────────────────────────────────

@app.post(
    "/process-audio",
    summary="Process audio → MIDI",
    response_description="MIDI file produced by the full ML pipeline",
)
async def process_audio(
    file: UploadFile = File(..., description="Audio file to process (.mp3 / .wav / .flac)"),
    instrument: str = Form(default="Piano", description="Target instrument (Piano, Guitar, Violin, …)"),
):
    """
    Accepts an audio upload, runs the three-stage ML pipeline, and returns the
    resulting MIDI file as a downloadable attachment.

    Stages:
      1. **prepare_for_llm.py** — stem separation + instrument detection + Basic Pitch
      2. **expert_arranger.py** — music-theory-aware arrangement
      3. **to_midi.py**         — MIDI assembly with humanisation
    """
    job_id = str(uuid.uuid4())

    # ── 1. Save uploaded file ─────────────────────────────────────────────────
    suffix = Path(file.filename).suffix or ".mp3"
    audio_path  = UPLOADS_DIR / f"{job_id}{suffix}"
    llm_json    = OUTPUTS_DIR / f"llm_input_{job_id}.json"
    arranged    = OUTPUTS_DIR / f"arranged_{job_id}.json"
    midi_output = OUTPUTS_DIR / f"output_{job_id}.mid"

    try:
        audio_bytes = await file.read()
        audio_path.write_bytes(audio_bytes)
        print(f"[API] Saved upload → {audio_path}  ({len(audio_bytes):,} bytes)")

        # ── 2. prepare_for_llm ────────────────────────────────────────────────
        run_step(
            [
                sys.executable, "prepare_for_llm.py",
                "--audio",      str(audio_path),
                "--output",     str(llm_json),
                "--instrument", instrument,
                "--skip-demucs",   # skip stem separation in server mode for speed
            ],
            "prepare_for_llm",
        )

        # ── 3. expert_arranger ────────────────────────────────────────────────
        run_step(
            [
                sys.executable, "expert_arranger.py",
                "--input",      str(llm_json),
                "--audio",      str(audio_path),
                "--output",     str(arranged),
                "--instrument", instrument,
            ],
            "expert_arranger",
        )

        # ── 4. to_midi ────────────────────────────────────────────────────────
        run_step(
            [
                sys.executable, "to_midi.py",
                "--input",      str(arranged),
                "--output",     str(midi_output),
                "--instrument", instrument,
            ],
            "to_midi",
        )

        if not midi_output.exists():
            raise HTTPException(status_code=500, detail="MIDI output not found after pipeline.")

        return FileResponse(
            path=str(midi_output),
            media_type="audio/midi",
            filename=f"output_{instrument.lower()}_{job_id[:8]}.mid",
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        # Clean up intermediate JSON files to save disk space
        for tmp in [llm_json, arranged]:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        # Note: we do NOT delete the audio upload here so callers can retry;
        # in production you'd add a background cleanup task.


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", summary="Health check")
def health():
    return {"status": "ok"}
