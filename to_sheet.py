import json
import argparse
from music21 import stream, note, tempo, metadata, environment
import music21

# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert transcription JSON → Sheet Music")
    parser.add_argument("--input",      default="transcription.json", help="Path to transcription JSON")
    parser.add_argument("--output",     default="sheet",              help="Output filename (no extension)")
    parser.add_argument("--instrument", default="Piano",              help="Instrument label on sheet")
    parser.add_argument("--title",      default="Transcription",      help="Title on sheet music")
    parser.add_argument("--format",     default="musicxml",           choices=["musicxml", "pdf", "png", "midi"],
                        help="Output format (pdf/png requires MuseScore installed)")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        notes_data = json.load(f)

    # ── Build Grand Staff ──
    score = stream.Score()
    score.metadata = metadata.Metadata()
    score.metadata.title  = args.title
    score.metadata.composer = f"Auto-transcribed for {args.instrument}"

    right_part = stream.PartStaff()
    right_part.id = 'right'
    right_part.insert(0, music21.clef.TrebleClef())
    
    left_part = stream.PartStaff()
    left_part.id = 'left'
    left_part.insert(0, music21.clef.BassClef())

    skipped = 0

    for entry in notes_data:
        note_name = entry.get("note")
        duration  = float(entry.get("duration", 0.5))
        start_time = float(entry.get("time", 0.0))
        hand = entry.get("hand", "right")

        if not note_name:
            skipped += 1
            continue

        try:
            # Duration in quarter notes (assuming 1 quarter = 0.5s at 120bpm)
            quarter_len = duration * 2.0   # seconds → quarter note length

            # Clamp to reasonable values
            quarter_len = max(0.25, min(quarter_len, 4.0))

            n = note.Note(note_name)
            n.quarterLength = quarter_len
            
            # Map time (seconds) to quarter notes roughly (120bpm -> 0.5s = 1 quarter)
            quarter_offset = start_time * 2.0
            
            if hand == "left":
                left_part.insert(quarter_offset, n)
            else:
                right_part.insert(quarter_offset, n)

        except Exception:
            skipped += 1
            continue

    score.insert(0, right_part)
    score.insert(0, left_part)

    # ── Export ──
    ext_map = {
        "musicxml": ".xml",
        "pdf":      ".pdf",
        "png":      ".png",
        "midi":     ".mid"
    }
    out_path = args.output + ext_map.get(args.format, ".xml")

    if args.format in ("pdf", "png"):
        # Requires MuseScore to be installed and configured
        us = environment.UserSettings()
        print(f"[INFO] Rendering {args.format.upper()} (needs MuseScore)...")
        score.write(args.format, fp=out_path)
    else:
        score.write(args.format, fp=out_path)

    print(f"[DONE] Sheet music saved → {out_path}  ({len(part)} notes, {skipped} skipped)")
    if skipped > 0:
        print(f"[WARN] {skipped} notes were skipped (invalid note names)")


if __name__ == "__main__":
    main()
