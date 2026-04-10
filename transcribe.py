import json
import argparse
import os
from openai import OpenAI

# ─── CONFIG ─────────────────────────────────────────────────────────────────

DEFAULT_INPUT  = "output2.json"
DEFAULT_OUTPUT = "transcription.json"
DEFAULT_MODEL  = "gpt-4o"

SYSTEM_PROMPT = """You are an expert music transcription and arrangement system.

You are given structured audio analysis data for an entire song.
Each entry represents a short time segment (~3 seconds).

Your goal is to convert the ENTIRE song into a playable sequence for ONE chosen instrument.

---

INPUT DATA DESCRIPTION:

Each segment contains:
- time_start, time_end
- detected_note (dominant pitch)
- dominant_frequency (Hz)
- chroma_mean (12 pitch intensities: C, C#, D, D#, E, F, F#, G, G#, A, A#, B)
- tempo (BPM)
- energy (loudness)

---

YOUR TASK:

Generate a complete musical transcription of the song as if it were played entirely on the target instrument.

---

IMPORTANT RULES:

1. MELODY EXTRACTION (CRITICAL)
   - Extract ONLY the dominant melodic line across the entire song.
   - Ignore background instruments and noise.
   - Maintain continuity between segments.

2. NOTE DETERMINATION
   - Use detected_note as the primary pitch.
   - Use chroma_mean to stabilize pitch (avoid jitter).
   - Use dominant_frequency for refinement if needed.

3. TIMING & RHYTHM
   - Convert time segments into musical durations using tempo.
   - Ensure rhythm is natural and consistent.
   - Merge consecutive identical notes into longer durations.

4. SMOOTHING
   - Remove noisy rapid note changes.
   - Prefer stable, musically meaningful transitions.

5. KEY CONSISTENCY
   - Try to stay within a consistent musical key.
   - Avoid random out-of-scale notes unless strongly supported.

6. INSTRUMENT ADAPTATION

   Adapt the output to be playable on the chosen instrument:

   - Guitar:
     - Use realistic note ranges
     - Avoid impossible jumps
     - Prefer stepwise motion

   - Piano:
     - Allow wider range
     - Smooth phrasing

   - Flute:
     - Single-note only
     - Avoid abrupt jumps

7. OUTPUT FORMAT

Return ONLY valid JSON — an array of note objects.
No prose, no explanation, no markdown.

[
  { "time": 0.0, "note": "C4", "duration": 0.5 },
  { "time": 0.5, "note": "D4", "duration": 0.5 },
  { "time": 1.0, "note": "E4", "duration": 1.0 }
]

8. FULL SONG REQUIREMENT

- Cover the entire duration of the song.
- Ensure continuity from start to end.
- No missing sections.

9. MUSICAL QUALITY

- The result should sound like a real playable version of the song.
- Avoid chaotic or random note sequences.
- Prioritize musicality over strict data fidelity.

---

FINAL INSTRUCTION:

Generate a complete, smooth, musically coherent transcription of the entire song for the chosen instrument.
Return ONLY the JSON array. Nothing else."""


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def load_analysis(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def build_user_message(data: dict, instrument: str) -> str:
    segments = data["segments"]

    lines = [
        f"TARGET INSTRUMENT: {instrument}",
        f"Total duration: {data['duration']:.2f}s | Segments: {data['num_segments']}",
        "",
        "SEGMENT DATA:",
        ""
    ]

    for seg in segments:
        line = (
            f"[{seg['time_start']:.1f}s–{seg['time_end']:.1f}s] "
            f"note={seg.get('detected_note') or '?'} | "
            f"freq={seg.get('dominant_frequency') or '?'}Hz | "
            f"tempo={seg.get('tempo', '?')}bpm | "
            f"energy={seg.get('energy', '?')} | "
            f"chroma={seg.get('chroma_mean') or '?'}"
        )
        lines.append(line)

    return "\n".join(lines)


def call_llm(client: OpenAI, user_message: str, model: str) -> str:
    print(f"[INFO] Sending to {model}...")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message}
        ],
        temperature=0.3,     # Low temp = more deterministic / musical
    )
    return response.choices[0].message.content.strip()


def parse_and_save(raw: str, output_path: str):
    # Strip markdown code fences if LLM adds them
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1])

    try:
        notes = json.loads(raw)
        with open(output_path, "w") as f:
            json.dump(notes, f, indent=2)
        print(f"[DONE] Transcription saved → {output_path}  ({len(notes)} notes)")
    except json.JSONDecodeError as e:
        # Fallback: save raw text
        raw_path = output_path.replace(".json", "_raw.txt")
        with open(raw_path, "w") as f:
            f.write(raw)
        print(f"[WARN] Could not parse JSON ({e}). Raw output saved → {raw_path}")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Transcribe instrument analysis JSON to note sequence")
    parser.add_argument("--input",      default=DEFAULT_INPUT,  help="Path to inference output JSON")
    parser.add_argument("--output",     default=DEFAULT_OUTPUT, help="Path to save transcription JSON")
    parser.add_argument("--instrument", default="Piano",        help="Target instrument (e.g., Piano, Guitar, Flute)")
    parser.add_argument("--model",      default=DEFAULT_MODEL,  help="OpenAI model to use")
    args = parser.parse_args()

    # ── API Key ──
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Missing OPENAI_API_KEY environment variable.\n"
            "Set it with: $env:OPENAI_API_KEY = 'sk-...'"
        )

    client = OpenAI(api_key=api_key)

    # ── Load & Send ──
    print(f"[INFO] Loading {args.input}...")
    data = load_analysis(args.input)

    user_message = build_user_message(data, args.instrument)

    raw = call_llm(client, user_message, args.model)

    # ── Save ──
    parse_and_save(raw, args.output)


if __name__ == "__main__":
    main()
