import json
import argparse
import pretty_midi
import random

# ─── NOTE NAME → MIDI NUMBER ─────────────────────────────────────────────────

NOTE_MAP = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
            "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
            "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}

INSTRUMENT_PROGRAMS = {
    "Piano":   0,    # Acoustic Grand Piano
    "Guitar":  24,   # Acoustic Guitar (nylon)
    "Acoustic Electric Guitar": 25, # Acoustic Guitar (steel)
    "Electric Guitar": 27, # Electric Guitar (clean)
    "Flute":   73,   # Flute
    "Violin":  40,   # Violin
    "Trumpet": 56,   # Trumpet
    "Organ":   19,   # Church Organ
    "Sitar":   104,  # Sitar
}

def note_name_to_midi(name: str) -> int | None:
    """Convert note name like 'C4', 'F#3' to MIDI number."""
    if not name:
        return None
    for i, ch in enumerate(name):
        if ch.isdigit() or (ch == '-' and i > 0):
            pitch_part = name[:i]
            octave = int(name[i:])
            if pitch_part in NOTE_MAP:
                return (octave + 1) * 12 + NOTE_MAP[pitch_part]
    return None


def apply_todd_phrasing(notes, tempo):
    """
    Todd (1992): Phrase-end ritardando and phrase-internal acceleration.
    - Last 3-4 notes before each phrase_start tag get duration stretched
      with a cubic deceleration curve (ritardando).
    - Notes approaching the highest-velocity (climax) note per phrase
      get their timing nudged slightly early (-5ms to -8ms).
    - Last note before phrase_start gets +0.3s additional decay.
    """
    if not notes:
        return notes

    # Find phrase boundaries
    phrase_end_indices = []
    for i, n in enumerate(notes):
        # A note is a phrase end if the NEXT note has phrase_start=True
        if i + 1 < len(notes) and notes[i + 1].get('phrase_start', False):
            phrase_end_indices.append(i)
    phrase_end_indices.append(len(notes) - 1)  # Final note is always a phrase end

    sec_per_beat = 60.0 / max(tempo, 1)

    # Build phrase groups for climax detection
    phrase_groups = []
    start_idx = 0
    for end_idx in phrase_end_indices:
        phrase_groups.append(notes[start_idx:end_idx + 1])
        start_idx = end_idx + 1

    for phrase in phrase_groups:
        if not phrase:
            continue

        # ── Ritardando: stretch final 3-4 notes ──
        rit_multipliers = [1.18, 1.14, 1.08, 1.0]  # oldest-to-newest
        rit_window = phrase[-4:] if len(phrase) >= 4 else phrase
        for i, note in enumerate(rit_window):
            mult = rit_multipliers[max(0, 4 - len(rit_window) + i)]
            note['_rit_mult'] = mult

        # ── Extended phrase-end decay (+0.3s on final note) ──
        phrase[-1]['_phrase_end_decay'] = True

        # ── Phrase-internal acceleration toward climax ──
        if len(phrase) > 4:
            climax_idx = max(range(len(phrase)), key=lambda i: phrase[i].get('velocity', 0))
            for i in range(max(0, climax_idx - 4), climax_idx):
                phrase[i]['_pre_climax_nudge'] = True  # Rush timing: -6ms

    return notes


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert transcription JSON → MIDI")
    parser.add_argument("--input",      default="transcription.json", help="Path to transcription JSON")
    parser.add_argument("--output",     default="output.mid",         help="Path to save .mid file")
    parser.add_argument("--instrument", default="Piano",              help="Instrument name for MIDI program")
    parser.add_argument("--tempo",      type=float, default=120.0,    help="BPM (default: 120)")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        notes = json.load(f)

    # ── Sort chronologically and apply Todd phrasing ──
    notes.sort(key=lambda x: x['time'])
    notes = apply_todd_phrasing(notes, args.tempo)

    # ── Setup MIDI ──
    midi = pretty_midi.PrettyMIDI(initial_tempo=args.tempo)
    program = INSTRUMENT_PROGRAMS.get(args.instrument, 0)
    instrument = pretty_midi.Instrument(program=program, name=args.instrument)

    skipped = 0
    for entry in notes:
        # Prefer direct midi field (updated by voice leading optimizer)
        midi_num = entry.get("midi")
        if midi_num is None:
            midi_num = note_name_to_midi(entry.get("note"))
        if midi_num is None:
            skipped += 1
            continue
        midi_num = int(midi_num)

        start    = float(entry["time"])
        duration = float(entry["duration"])
        role     = entry.get("role", "harmony")

        # ── Todd (1992): Apply ritardando duration stretch ──
        rit_mult = entry.get('_rit_mult', 1.0)
        duration *= rit_mult

        # ── Todd (1992): Pre-climax timing nudge (rush toward peak) ──
        if entry.get('_pre_climax_nudge', False):
            start -= 0.006  # Rush 6ms early
        
        # ── Instrument-Specific Timing Micro-Offsets ──
        offset_ms = 0
        if args.instrument == "Piano":
            offset_ms = random.uniform(-10, 10)
        elif "Guitar" in args.instrument:
            offset_ms = -10 if role == 'bass' else 15
        elif args.instrument in ["Sitar", "Violin", "Flute"]:
            offset_ms = 25 if role == 'melody' else 10
            
        start += (offset_ms / 1000.0)

        # ── Acoustic Decay by Musical Role + Todd phrase-end extension ──
        if role == 'melody':
            decay_tail = 0.6
        elif role == 'bass':
            decay_tail = 0.4
        else:
            decay_tail = 0.3

        if entry.get('_phrase_end_decay', False):
            decay_tail += 0.3  # Extra ring-out on phrase endings
            
        end = start + duration + decay_tail

        raw_vel = float(entry.get("velocity", 80)) # Might be 0-1.0 or 0-127
        if raw_vel <= 1.0: raw_vel *= 127
            
        # Map dynamic velocity ranges
        norm_v = min(1.0, raw_vel / 127.0)
        if role == 'melody':
            base_vel = int(90 + (norm_v * 20)) # 90-110
        elif role == 'bass':
            base_vel = int(60 + (norm_v * 20)) # 60-80
        else:
            base_vel = int(50 + (norm_v * 20)) # 50-70
            
        # Humanize: Random Jitter
        vel = base_vel + random.randint(-10, 10)
        
        # Tempo-Aware Beat Accents
        sec_per_beat = 60.0 / args.tempo
        beat_idx = round(start / sec_per_beat)
        is_on_grid = abs(start - (beat_idx * sec_per_beat)) < 0.05
        
        if is_on_grid:
            if beat_idx % 4 == 0:     # Beat 1
                vel += 8
            elif beat_idx % 4 == 2:   # Beat 3
                vel += 3 if args.tempo > 120 else 8
        else: # Off-beat
            vel -= 5
            
        # Phrase start re-attack
        if entry.get("phrase_start", False):
            vel += 5
            
        # Clamp velocity bounds
        vel = max(1, min(127, vel))

        note = pretty_midi.Note(
            velocity=vel,
            pitch=midi_num,
            start=start,
            end=end
        )
        instrument.notes.append(note)

    midi.instruments.append(instrument)
    midi.write(args.output)

    print(f"[DONE] MIDI saved -> {args.output}  ({len(instrument.notes)} notes, {skipped} skipped)")


if __name__ == "__main__":
    main()
