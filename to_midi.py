"""
to_midi.py — MelodAI MIDI Generator  v2.0
==========================================
Converts arranged.json → .mid with a full expression engine:

  • Role-stratified velocity ranges
  • Gaussian timing humanisation per role
  • Idiomatic articulation (melody legato, harmony staccato-ish)
  • Todd (1992) phrase ritardando + velocity arc
  • Beat-accent grid awareness
  • Guitar pluck envelope shaping
  • Instrument-specific timing micro-offsets

All function signatures and output format unchanged.
"""

import json
import argparse
import random
import math
import pretty_midi

# ─── NOTE NAME → MIDI NUMBER ─────────────────────────────────────────────────

NOTE_MAP = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
            "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
            "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}

INSTRUMENT_PROGRAMS = {
    "Piano":                    0,    # Acoustic Grand Piano
    "Guitar":                   24,   # Acoustic Guitar (nylon)
    "Acoustic Electric Guitar": 25,   # Acoustic Guitar (steel)
    "Electric Guitar":          27,   # Electric Guitar (clean)
    "Flute":                    73,   # Flute
    "Violin":                   40,   # Violin
    "Trumpet":                  56,   # Trumpet
    "Organ":                    19,   # Church Organ
    "Sitar":                    104,  # Sitar
}


def note_name_to_midi(name: str) -> int | None:
    """Convert note name like 'C4', 'F#3' to MIDI number."""
    if not name:
        return None
    for i, ch in enumerate(name):
        if ch.isdigit() or (ch == '-' and i > 0):
            pitch_part = name[:i]
            octave     = int(name[i:])
            if pitch_part in NOTE_MAP:
                return (octave + 1) * 12 + NOTE_MAP[pitch_part]
    return None


# ─── TODD PHRASING ───────────────────────────────────────────────────────────

def apply_todd_phrasing(notes, tempo):
    """
    Todd (1992): phrase-end ritardando, pre-climax acceleration,
    + symmetric velocity arc (crescendo → climax → decrescendo).

    V2 improvements:
      • Velocity arc is applied here if '_phrase_vel_scale' was NOT already set
        by the arranger (backward-compat).
      • Ritardando curve is cubic rather than linear for a more musical feel.
      • Final note of each phrase gets a longer decay tail flag.
      • Pre-climax nudge is extended to 5 notes (was 4).
    """
    if not notes:
        return notes

    # Locate phrase boundaries
    phrase_end_indices = []
    for i, n in enumerate(notes):
        if i + 1 < len(notes) and notes[i + 1].get('phrase_start', False):
            phrase_end_indices.append(i)
    phrase_end_indices.append(len(notes) - 1)

    phrase_groups = []
    start_idx = 0
    for end_idx in phrase_end_indices:
        phrase_groups.append(notes[start_idx:end_idx + 1])
        start_idx = end_idx + 1

    for phrase in phrase_groups:
        if not phrase:
            continue

        # ── Ritardando: stretch final 4 notes with cubic deceleration ──
        rit_window = phrase[-4:] if len(phrase) >= 4 else phrase
        n_rit      = len(rit_window)
        for i, pn in enumerate(rit_window):
            # cubic: ratio → 1.0 at start, 1.22 at end
            t    = i / max(n_rit - 1, 1)
            mult = 1.0 + 0.22 * (t ** 3)
            pn['_rit_mult'] = round(mult, 3)

        # ── Extended phrase-end decay ──
        phrase[-1]['_phrase_end_decay'] = True

        # ── Pre-climax acceleration (rush 5 notes toward climax) ──
        if len(phrase) > 5:
            climax_idx = max(range(len(phrase)),
                             key=lambda i: phrase[i].get('velocity', 0))
            for i in range(max(0, climax_idx - 5), climax_idx):
                phrase[i]['_pre_climax_nudge'] = True

        # ── Velocity arc (only if arranger did NOT already apply shape) ──
        if not any(pn.get('_phrase_vel_scale') for pn in phrase):
            climax_idx = max(range(len(phrase)),
                             key=lambda i: phrase[i].get('velocity', 0))
            n_ph = len(phrase)
            for i, pn in enumerate(phrase):
                if i <= climax_idx:
                    frac  = i / max(climax_idx, 1)
                    scale = 0.78 + 0.32 * frac
                else:
                    frac  = (i - climax_idx) / max(n_ph - climax_idx - 1, 1)
                    scale = 1.10 - 0.38 * frac
                pn['_phrase_vel_scale'] = round(scale, 3)

    return notes


# ─── ARTICULATION PROFILES ───────────────────────────────────────────────────

def _articulation_duration(role: str, duration: float, instrument: str) -> float:
    """
    Return articulated note duration:
      Melody  → legato  (0.96 × raw)
      Bass    → normal  (0.92 × raw)
      Harmony → shorter (0.70 × raw, except Guitar where it is 0.60)
    Guitar-specific: all notes get pluck shaping (fast attack, quick release).
    """
    if "Guitar" in instrument:
        # Guitar pluck: melody slightly longer, bass a bit longer still
        if role == 'melody':  return max(0.05, duration * 0.85)
        if role == 'bass':    return max(0.05, duration * 0.75)
        return max(0.05, duration * 0.55)    # chord strum fragments

    if role == 'melody':   return max(0.05, duration * 0.96)
    if role == 'bass':     return max(0.05, duration * 0.92)
    return max(0.05, duration * 0.70)        # harmony


def _timing_jitter(role: str, instrument: str) -> float:
    """
    Return timing offset in seconds (Gaussian).
    Values calibrated to feel human without feeling sloppy.
    """
    sigma_map = {
        'melody':  0.006,    # ≈6ms  — expressive rubato
        'bass':    0.004,    # ≈4ms  — rock-solid with slight push
        'harmony': 0.009,    # ≈9ms  — scatter around beat
    }
    sigma = sigma_map.get(role, 0.007)

    # Strings / wind: add a tiny extra variation for breath/bow feel
    if instrument in ("Violin", "Flute", "Trumpet", "Sitar"):
        sigma += 0.003

    return random.gauss(0, sigma)


# ─── VELOCITY ENGINE ─────────────────────────────────────────────────────────

def _compute_velocity(entry: dict, role: str, tempo: float,
                      sec_per_beat: float, instrument: str) -> int:
    """
    Full velocity pipeline:
      1. Role-stratified base range
      2. Todd phrase-dynamics arc
      3. Beat-grid accent
      4. Phrase-start re-attack
      5. Gaussian jitter
      6. Guitar pluck boost on downbeats
    Returns clamped [1, 127].
    """
    raw_vel = float(entry.get("velocity", 80))
    if raw_vel <= 1.0:
        raw_vel *= 127.0
    norm_v = min(1.0, raw_vel / 127.0)

    # 1. Role-stratified range
    if   role == 'melody':   base_vel = 88 + norm_v * 20    # 88–108
    elif role == 'bass':     base_vel = 62 + norm_v * 18    # 62–80
    else:                    base_vel = 44 + norm_v * 18    # 44–62

    # 2. Todd phrase arc
    phrase_scale = entry.get('_phrase_vel_scale', 1.0)
    base_vel    *= phrase_scale

    # 3. Beat-grid accent
    start    = float(entry['time'])
    beat_idx = round(start / sec_per_beat) if sec_per_beat > 0 else 0
    on_grid  = abs(start - beat_idx * sec_per_beat) < 0.05
    if on_grid:
        if   beat_idx % 4 == 0: base_vel += 8   # Beat 1 — strong
        elif beat_idx % 4 == 2: base_vel += (3 if tempo > 120 else 8)
    else:
        base_vel -= 5    # off-beat notes slightly softer

    # 4. Phrase-start re-attack
    if entry.get('phrase_start', False):
        base_vel += 6

    # 5. Guitar downbeat pluck accentuation
    if "Guitar" in instrument and on_grid and beat_idx % 2 == 0:
        base_vel += 4

    # 6. Gaussian jitter per role
    jitter_sigma = {'melody': 5, 'bass': 4, 'harmony': 7}.get(role, 5)
    base_vel    += random.gauss(0, jitter_sigma)

    return max(1, min(127, int(round(base_vel))))


# ─── DECAY TAIL BY ROLE ──────────────────────────────────────────────────────

def _decay_tail(role: str, instrument: str, phrase_end: bool) -> float:
    """
    Acoustic ring-out tail added to note duration.
    Guitar: shorter (plucked), Violin: longer (bowed).
    """
    tails = {
        'melody':  {'base': 0.55, 'Guitar': 0.25, 'Violin': 0.80, 'Flute': 0.65},
        'bass':    {'base': 0.35, 'Guitar': 0.18, 'Violin': 0.45, 'Flute': 0.35},
        'harmony': {'base': 0.25, 'Guitar': 0.12, 'Violin': 0.35, 'Flute': 0.28},
    }
    role_tails = tails.get(role, tails['harmony'])
    tail = role_tails.get(instrument, role_tails['base'])
    if phrase_end:
        tail += 0.30    # extra decay at phrase endings
    return tail


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

    # ── Sort and apply Todd phrasing ──
    notes.sort(key=lambda x: x['time'])
    notes = apply_todd_phrasing(notes, args.tempo)

    # ── Setup MIDI ──
    midi       = pretty_midi.PrettyMIDI(initial_tempo=args.tempo)
    program    = INSTRUMENT_PROGRAMS.get(args.instrument, 0)
    instrument = pretty_midi.Instrument(program=program, name=args.instrument)

    sec_per_beat = 60.0 / max(args.tempo, 1.0)
    skipped      = 0

    for entry in notes:
        midi_num = entry.get("midi")
        if midi_num is None:
            midi_num = note_name_to_midi(entry.get("note"))
        if midi_num is None:
            skipped += 1;  continue
        midi_num = int(midi_num)

        role     = entry.get("role", "harmony")
        start    = float(entry["time"])
        duration = float(entry["duration"])

        # ── Todd ritardando ──
        rit_mult  = entry.get('_rit_mult', 1.0)
        duration *= rit_mult

        # ── Articulation shaping ──
        duration = _articulation_duration(role, duration, args.instrument)

        # ── Pre-climax time nudge ──
        if entry.get('_pre_climax_nudge', False):
            start -= 0.006

        # ── Timing jitter ──
        start += _timing_jitter(role, args.instrument)
        start  = max(0.0, start)

        # ── Decay tail + phrase-end extension ──
        tail  = _decay_tail(role, args.instrument,
                            phrase_end=entry.get('_phrase_end_decay', False))
        end   = start + duration + tail

        # Guard against zero-length notes
        if end <= start:
            end = start + 0.05

        # ── Velocity ──
        vel = _compute_velocity(entry, role, args.tempo, sec_per_beat, args.instrument)

        note = pretty_midi.Note(velocity=vel, pitch=midi_num, start=start, end=end)
        instrument.notes.append(note)

    midi.instruments.append(instrument)
    midi.write(args.output)

    print(f"[DONE] MIDI saved -> {args.output}  "
          f"({len(instrument.notes)} notes, {skipped} skipped)")


if __name__ == "__main__":
    main()
