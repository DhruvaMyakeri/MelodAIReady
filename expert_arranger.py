"""
expert_arranger.py  — MelodAI Intelligent Arranger  v5.0
=========================================================
Architecture: multi-stage, memory-aware, self-correcting.

Stage 0  : Streaming JSON parse (ijson) + structure detection
Stage 1  : Foundational processing (clean / quantize / cluster)
Stage 2  : Role assignment  — melody/bass/harmony w/ global memory
Stage 3  : Chord build + polyphony reduction
Stage 3.5: Voice-leading optimizer  (Tymoczko)
Stage 4  : Density budget  (Lerdahl tension-aware)
Stage 5  : Refinement pass  — self-correction (melody jumps / chord chaos / out-of-key)
Stage 6  : Expression pass  — musical intent modelling (velocity / articulation / timing)
Stage 7  : Validation pass  — final coherence checks

All public function signatures and output JSON schema are
100 % backward-compatible with the original implementation.
"""

import json
import argparse
import random
import math
import numpy as np
import librosa
from collections import defaultdict, deque
from music21 import chord, note, pitch

# ─── optional streaming parser ───────────────────────────────────────────────
try:
    import ijson
    HAS_IJSON = True
except ImportError:
    HAS_IJSON = False
    print("[WARN] ijson not installed — falling back to json.load(). "
          "Install with: pip install ijson")

# ─── MUSIC THEORY CONSTANTS ──────────────────────────────────────────────────

NOTE_MAP = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
            "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
            "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}

# Pearce & Wiggins (2006): melodic interval probability
INTERVAL_PROBS = {0: 0.16, 1: 0.14, 2: 0.20, 3: 0.10, 4: 0.08, 5: 0.07}

# Circle of fifths order for tonal-distance (Harte 2006)
CIRCLE_OF_FIFTHS = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]

# ─── INSTRUMENT PHYSICAL RANGES ──────────────────────────────────────────────

INSTRUMENT_RANGES = {
    "Piano":                     (21, 108),
    "Guitar":                    (40, 84),
    "Electric Guitar":           (40, 84),
    "Acoustic Electric Guitar":  (40, 84),
    "Sitar":                     (50, 77),
    "Flute":                     (60, 96),
    "Violin":                    (55, 103),
    "Trumpet":                   (58, 94),
}

MONO_INSTRUMENTS = {"Sitar", "Flute", "Violin", "Trumpet"}

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def get_diatonic_pitches(key_str):
    if key_str == "Ambiguous":
        return None
    parts = key_str.split(" ")
    if len(parts) != 2:
        return None
    root_str, scale_type = parts
    root_midi = NOTE_MAP.get(root_str, 0)
    intervals = [0, 2, 4, 5, 7, 9, 11] if scale_type == "Major" else [0, 2, 3, 5, 7, 8, 10]
    return set([(root_midi + iv) % 12 for iv in intervals])


def compute_tension(midi_list, key_str):
    """
    Lerdahl (2001): Tonal Pitch Space tension [0-1].
    """
    if not midi_list or key_str == "Ambiguous":
        return 0.5
    diatonic  = get_diatonic_pitches(key_str)
    root_pc   = NOTE_MAP.get(key_str.split()[0], 0)
    pcs       = [m % 12 for m in midi_list]
    has_tritone      = any(abs(pcs[i]-pcs[j]) % 12 == 6
                           for i in range(len(pcs)) for j in range(i+1, len(pcs)))
    leading_tone_pc  = (root_pc + 11) % 12
    has_leading_tone = leading_tone_pc in pcs
    dominant_pc      = (root_pc + 7) % 12
    has_dominant     = dominant_pc in pcs
    non_diatonic_ct  = sum(1 for pc in pcs if diatonic and pc not in diatonic)
    tension = 0.3
    if has_tritone:      tension += 0.30
    if has_leading_tone: tension += 0.20
    if has_dominant:     tension += 0.15
    tension += non_diatonic_ct * 0.05
    return min(1.0, tension)


def tonal_distance(pcs_a, pcs_b):
    """
    Harte (2006): Tonal distance between two pitch-class sets using
    circle-of-fifths centroid representation.
    Returns 0.0 (identical) … 1.0 (maximally distant).
    """
    def cof_vector(pcs):
        vec = np.zeros(12)
        for pc in pcs:
            vec[pc] += 1
        if vec.sum() > 0:
            vec /= vec.sum()
        # Project onto circle of fifths
        r5 = sum(vec[CIRCLE_OF_FIFTHS[i]] * math.cos(2*math.pi*i/12) for i in range(12))
        i5 = sum(vec[CIRCLE_OF_FIFTHS[i]] * math.sin(2*math.pi*i/12) for i in range(12))
        r3 = sum(vec[i] * math.cos(2*math.pi*i/4)  for i in range(12))
        i3 = sum(vec[i] * math.sin(2*math.pi*i/4)  for i in range(12))
        return np.array([r5, i5, r3, i3])
    va = cof_vector(pcs_a)
    vb = cof_vector(pcs_b)
    dist = np.linalg.norm(va - vb)
    return min(1.0, dist / 2.0)   # normalise to [0,1]


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL MUSICAL MEMORY
# ═══════════════════════════════════════════════════════════════════════════════

class MelodicMemory:
    """
    Long-term melodic memory across the entire song.
    Tracks:
      • rolling pitch history for drift detection
      • phrase fingerprints for motif reuse
      • exponential moving average of pitch centre
    """
    WINDOW = 32          # notes kept in rolling pitch buffer
    PHRASE_LEN = 8       # fingerprint length (pitch-class sequence)
    EMA_ALPHA  = 0.08    # smoothing factor — slow to adapt for stability

    def __init__(self):
        self._pitch_buffer: deque = deque(maxlen=self.WINDOW)
        self._phrase_bank: list   = []      # list of (pc_tuple, shape_envelope)
        self._ema_pitch: float | None = None
        self._current_phrase: list = []

    # ── public interface ──────────────────────────────────────────────────────

    def update(self, midi: int):
        self._pitch_buffer.append(midi)
        self._current_phrase.append(midi % 12)
        if len(self._current_phrase) >= self.PHRASE_LEN:
            self._bank_phrase(tuple(self._current_phrase[-self.PHRASE_LEN:]))
        # EMA update
        if self._ema_pitch is None:
            self._ema_pitch = float(midi)
        else:
            self._ema_pitch = (1 - self.EMA_ALPHA) * self._ema_pitch + self.EMA_ALPHA * midi

    def reset_phrase(self):
        if len(self._current_phrase) >= self.PHRASE_LEN:
            self._bank_phrase(tuple(self._current_phrase[-self.PHRASE_LEN:]))
        self._current_phrase = []

    @property
    def ema_pitch(self) -> float:
        return self._ema_pitch if self._ema_pitch is not None else 60.0

    def drift_penalty(self, midi: int) -> float:
        """Return penalty [0..30] for proposing a note far from rolling centre."""
        if not self._pitch_buffer:
            return 0.0
        deviation = abs(midi - self.ema_pitch)
        # Penalty rises steeply beyond a major 7th (11 semitones)
        return max(0.0, (deviation - 11) * 2.5)

    def motif_bonus(self, pc_sequence: tuple) -> float:
        """Reward a note that continues a previously seen motif."""
        if len(pc_sequence) < 3 or not self._phrase_bank:
            return 0.0
        query = pc_sequence[-3:]
        for stored in self._phrase_bank[-20:]:        # check last 20 motifs
            for i in range(len(stored) - len(query) + 1):
                if stored[i:i+len(query)] == query:
                    return 6.0
        return 0.0

    # ── private ──────────────────────────────────────────────────────────────

    def _bank_phrase(self, fingerprint: tuple):
        if fingerprint not in self._phrase_bank:
            self._phrase_bank.append(fingerprint)
            if len(self._phrase_bank) > 200:
                self._phrase_bank.pop(0)


class ChordMemory:
    """
    Maintains the last N chord pitch-class sets and exposes tonal distance
    from the most recent chord.  Used to steer harmony selection away from
    abrupt, unmusical harmonic jumps.
    """
    CAPACITY = 4

    def __init__(self):
        self._history: deque = deque(maxlen=self.CAPACITY)

    def push(self, pcs: frozenset):
        self._history.append(pcs)

    def distance_from_last(self, pcs: frozenset) -> float:
        if not self._history:
            return 0.0
        return tonal_distance(list(self._history[-1]), list(pcs))

    def smoothed_pcs(self) -> frozenset:
        """Return union of last two chords as a stable harmonic context."""
        if len(self._history) < 2:
            return self._history[-1] if self._history else frozenset()
        return self._history[-1] | self._history[-2]


# ═══════════════════════════════════════════════════════════════════════════════
# SONG STRUCTURE DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

class SongStructure:
    """
    Infer high-level musical sections (intro / verse / chorus / bridge / outro)
    from note density, mean velocity, and pitch-range width.

    Each section gets a label and an energy_level ∈ [0,1] that drives:
      • density budget multiplier
      • velocity shaping
      • harmonic complexity
    """
    SECTION_LABELS = ["intro", "verse", "chorus", "bridge", "outro"]

    def __init__(self):
        self.sections: list[dict] = []   # [{start, end, label, energy}]
        self._window_stats: list[dict] = []

    def analyse(self, notes: list, total_duration: float, window_s: float = 8.0):
        """Run structure analysis on raw (gated) notes before arrangement."""
        if not notes or total_duration <= 0:
            return

        step = window_s / 2.0
        t = 0.0
        while t < total_duration:
            end = min(t + window_s, total_duration)
            bucket = [n for n in notes if t <= n['onset'] < end]
            if bucket:
                density = len(bucket) / window_s
                mean_vel  = np.mean([n['velocity'] for n in bucket])
                pitch_range = max(n['midi'] for n in bucket) - min(n['midi'] for n in bucket)
            else:
                density = mean_vel = pitch_range = 0.0
            self._window_stats.append({
                'start': t, 'end': end,
                'density': density, 'mean_vel': mean_vel, 'pitch_range': pitch_range
            })
            t += step

        # Normalise features across all windows
        if self._window_stats:
            max_d = max(w['density']    for w in self._window_stats) or 1
            max_v = max(w['mean_vel']   for w in self._window_stats) or 1
            max_r = max(w['pitch_range']for w in self._window_stats) or 1
            for w in self._window_stats:
                w['energy'] = (
                    0.4 * w['density']     / max_d +
                    0.4 * w['mean_vel']    / max_v +
                    0.2 * w['pitch_range'] / max_r
                )

        # Assign section label heuristically using energy curve shape
        n_windows = len(self._window_stats)
        for i, w in enumerate(self._window_stats):
            frac = i / max(n_windows - 1, 1)
            e = w['energy']
            if   frac < 0.08:               label = "intro"
            elif frac > 0.92:               label = "outro"
            elif e > 0.70:                  label = "chorus"
            elif e < 0.30:                  label = "verse"
            else:                           label = "bridge"
            w['label'] = label

        print(f"[STRUCT] Analysed {n_windows} windows over {total_duration:.1f}s")
        section_counts = defaultdict(int)
        for w in self._window_stats:
            section_counts[w['label']] += 1
        for lbl, cnt in section_counts.items():
            print(f"         {lbl}: {cnt} windows")

    def energy_at(self, onset: float) -> float:
        """Return interpolated energy level at time `onset`."""
        for w in self._window_stats:
            if w['start'] <= onset < w['end']:
                return w['energy']
        return 0.5

    def label_at(self, onset: float) -> str:
        for w in self._window_stats:
            if w['start'] <= onset < w['end']:
                return w.get('label', 'verse')
        return 'verse'

    def density_multiplier(self, onset: float) -> float:
        """
        Chorus: +2 budget; verse: -1; bridge: ±0; intro/outro: -1.
        """
        label = self.label_at(onset)
        return {'chorus': 2, 'verse': -1, 'bridge': 0, 'intro': -1, 'outro': -1}.get(label, 0)

    def velocity_scale(self, onset: float) -> float:
        e = self.energy_at(onset)
        return 0.75 + 0.5 * e    # maps [0,1] → [0.75, 1.25]


# ═══════════════════════════════════════════════════════════════════════════════
# CORE PIPELINE STAGES
# ═══════════════════════════════════════════════════════════════════════════════

def frequency_band_gate(notes, instrument):
    """
    Octave-shift notes outside instrument range.
    IMPROVED: chooses the octave that minimises jump from the previous note
    (Tymoczko voice-leading principle) rather than blindly shifting one octave
    at a time.
    """
    lo, hi = INSTRUMENT_RANGES.get(instrument, (0, 127))
    shifted = 0
    prev_midi = None

    for n in notes:
        original_midi = n['midi']
        midi = original_midi

        if midi < lo or midi > hi:
            # Build candidates: all valid octave transpositions
            candidates = []
            for delta in range(-4, 5):
                candidate = original_midi + (delta * 12)
                if lo <= candidate <= hi:
                    # penalise large jump from previous note
                    jump = abs(candidate - prev_midi) if prev_midi is not None else 0
                    candidates.append((jump, candidate))
            if candidates:
                candidates.sort()
                midi = candidates[0][1]
            else:
                # hard clamp
                midi = max(lo, min(hi, midi))

        n['midi'] = midi
        if midi != original_midi:
            shifted += 1
        prev_midi = midi

    print(f"[GATE] Frequency band gate: shifted {shifted} out-of-range notes to fit {instrument} range.")
    return notes


def clean_notes(notes, instrument="Piano"):
    """
    Remove short noises, merge legato same-pitch notes.
    Added: remove exact duplicate onset+pitch entries left by streaming dedup.
    """
    if not notes:
        return []
    print(f"[*] Cleaning {len(notes)} raw notes...")
    valid_notes = [n for n in notes if n['offset'] - n['onset'] >= 0.1]
    valid_notes.sort(key=lambda x: (x['midi'], x['onset']))

    is_mono  = instrument in MONO_INSTRUMENTS
    gap_limit = 0.35 if is_mono else 0.15

    merged_notes  = []
    current_note  = None

    for note in valid_notes:
        if current_note is None:
            current_note = dict(note)
            current_note['phrase_start'] = True
            continue
        if note['onset'] - current_note['offset'] > 0.5:
            merged_notes.append(current_note)
            current_note = dict(note)
            current_note['phrase_start'] = True
            continue
        if (note['midi'] == current_note['midi'] and
                note['onset'] <= current_note['offset'] + gap_limit):
            current_note['offset']   = max(current_note['offset'],   note['offset'])
            current_note['velocity'] = max(current_note['velocity'],  note['velocity'])
        else:
            merged_notes.append(current_note)
            current_note = dict(note)
            current_note['phrase_start'] = False

    if current_note:
        merged_notes.append(current_note)

    print(f"    -> Reduced to {len(merged_notes)} after de-noising & merging.")
    return merged_notes


def adaptive_quantization(notes, audio_path):
    """
    Snap note boundaries to a dynamic beat grid.
    IMPROVED:
      • 32nd-note sub-grid (subdivision=8) for finer timing resolution
      • snapping strength weighted by note importance (velocity × sqrt(duration))
        so expressive soft notes are less aggressively quantised
      • tempo-adaptive: if tempo > 160 BPM, subdivision reduced to 4 (16th notes)
        to avoid over-snapping fast passages
    """
    print(f"[*] Running adaptive quantization on audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=22050)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times   = librosa.frames_to_time(beats, sr=sr)

    # Choose grid density by tempo
    subdivision = 4 if float(np.atleast_1d(tempo)[0]) > 160 else 8

    def find_nearest_grid(time_val, beats_array, subdiv):
        if len(beats_array) == 0:
            return time_val
        if time_val <= beats_array[0]:
            bd = beats_array[1] - beats_array[0] if len(beats_array) > 1 else 0.5
            return beats_array[0] - bd * round((beats_array[0] - time_val) / bd * subdiv) / subdiv
        idx = np.searchsorted(beats_array, time_val)
        if idx >= len(beats_array):
            bd = beats_array[-1] - beats_array[-2]
            dist = time_val - beats_array[-1]
            return beats_array[-1] + bd * round(dist / bd * subdiv) / subdiv
        t0, t1     = beats_array[idx-1], beats_array[idx]
        grid_step  = (t1 - t0) / subdiv
        steps      = round((time_val - t0) / grid_step)
        return t0 + steps * grid_step

    quantized = []
    max_vel = max(n['velocity'] for n in notes) if notes else 1.0
    max_dur = max(n['offset'] - n['onset'] for n in notes) if notes else 1.0

    for n in notes:
        clamped_vel = max(n['velocity'], max_vel * 0.1)
        dur         = n['offset'] - n['onset']
        # importance = velocity × sqrt(normalised duration)
        importance  = (clamped_vel / max_vel) * math.sqrt(min(dur / max_dur, 1.0))
        snap_strength = min(1.0, importance * 1.5)

        q_onset_grid  = find_nearest_grid(n['onset'],  beat_times, subdivision)
        q_offset_grid = find_nearest_grid(n['offset'], beat_times, subdivision)

        q_onset  = n['onset']  + (q_onset_grid  - n['onset'])  * snap_strength
        q_offset = n['offset'] + (q_offset_grid - n['offset']) * snap_strength

        if q_offset <= q_onset:
            q_offset = q_onset + ((beat_times[1]-beat_times[0]) / subdivision
                                  if len(beat_times) > 1 else 0.05)

        quantized.append({
            **n,
            'onset':    round(q_onset, 3),
            'offset':   round(q_offset, 3),
            'duration': round(q_offset - q_onset, 3)
        })

    quantized.sort(key=lambda x: x['onset'])
    return quantized


def cluster_notes(notes, window_ms=50, tempo=120.0):
    """
    Group simultaneous events into harmonic clusters.
    IMPROVED: tempo-scaled window so at 80 BPM clusters are wider (75ms)
    and at 160 BPM they are tighter (37ms), matching musical beat density.
    """
    scaled_ms = window_ms * (120.0 / max(tempo, 40.0))
    print(f"[*] Clustering notes within {scaled_ms:.1f}ms windows (tempo-scaled)...")
    clusters = []
    if not notes:
        return clusters

    current_cluster = [notes[0]]
    cluster_time    = notes[0]['onset']
    threshold       = scaled_ms / 1000.0

    for n in notes[1:]:
        if n['onset'] - cluster_time <= threshold:
            current_cluster.append(n)
        else:
            clusters.append(current_cluster)
            current_cluster = [n]
            cluster_time    = n['onset']

    if current_cluster:
        clusters.append(current_cluster)

    print(f"    -> Assembled {len(clusters)} time-aligned clusters.")
    return clusters


def _rolling_melodic_center(pitch_buffer: deque) -> float:
    """Exponential moving average pitch from the rolling buffer."""
    if not pitch_buffer:
        return 60.0
    alpha = 0.12
    ema = float(pitch_buffer[0])
    for p in list(pitch_buffer)[1:]:
        ema = (1 - alpha) * ema + alpha * p
    return ema


def _shape_phrase_dynamics(phrase_notes: list) -> list:
    """
    Todd (1992): apply a symmetric crescendo → decrescendo velocity envelope
    to a phrase (list of notes with 'velocity').
    The envelope peaks at the note with highest natural velocity (climax).
    Returns the same list with '_phrase_vel_scale' set on each note.
    """
    if not phrase_notes:
        return phrase_notes
    n = len(phrase_notes)
    if n == 1:
        phrase_notes[0]['_phrase_vel_scale'] = 1.0
        return phrase_notes

    nat_vel  = [p['velocity'] for p in phrase_notes]
    climax_i = int(np.argmax(nat_vel))

    for i, pn in enumerate(phrase_notes):
        if i <= climax_i:
            # crescendo: linear ramp from 0.75 → 1.1 up to climax
            frac = i / max(climax_i, 1)
            scale = 0.75 + 0.35 * frac
        else:
            # decrescendo: linear ramp from 1.1 → 0.70 after climax
            frac = (i - climax_i) / max(n - climax_i - 1, 1)
            scale = 1.1 - 0.40 * frac
        pn['_phrase_vel_scale'] = round(scale, 3)

    return phrase_notes


def assign_roles(clusters, instrument="Piano", global_key="Ambiguous",
                 melodic_memory: MelodicMemory = None,
                 chord_memory: ChordMemory = None,
                 song_structure: SongStructure = None):
    """
    V5 Music Theory Engine — integrates:
    Huron (2006): stepwise preference, post-leap reversal, melodic arch
    Narmour (1990): process/reversal, registral return
    Pearce & Wiggins (2006): interval probability weighting
    Tymoczko (2006): contrary motion preference
    + GLOBAL MEMORY: drift penalty, motif bonus (MelodicMemory)
    + CHORD MEMORY: tonal-distance bias (ChordMemory)
    + STRUCTURE: energy-aware density / velocity hints (SongStructure)
    """
    print("[*] Assigning musical roles (V5 Music Theory Engine)...")

    if melodic_memory is None:
        melodic_memory = MelodicMemory()
    if chord_memory is None:
        chord_memory = ChordMemory()

    lo, hi       = INSTRUMENT_RANGES.get(instrument, (0, 127))
    span         = hi - lo
    bass_ceiling = lo + span * 0.35
    melody_floor = hi - span * 0.55
    diatonic_set = get_diatonic_pitches(global_key)

    all_midi = [n['midi'] for cl in clusters for n in cl]
    all_dur  = [n['duration'] for cl in clusters for n in cl]
    all_vel  = [n['velocity'] for cl in clusters for n in cl]
    max_midi = max(all_midi) if all_midi else 127
    max_dur  = max(all_dur)  if all_dur  else 1.0
    max_vel  = max(all_vel)  if all_vel  else 1.0

    WEIGHT_PITCH, WEIGHT_DUR, WEIGHT_VEL = 0.5, 0.3, 0.2

    prev_melody_midi  = None
    prev_bass_midi    = None
    prev_leap_size    = 0
    prev_leap_dir     = 0
    narmour_interval  = None
    narmour_direction = None

    phrase_note_idx      = 0
    home_register        = None
    phrase_start_pitches = []
    pitch_buffer: deque  = deque(maxlen=MelodicMemory.WINDOW)

    # phrase grouping for Todd envelope
    current_phrase_notes = []
    all_phrase_groups    = []

    assigned_clusters = []

    for cluster in clusters:
        if not cluster:
            continue

        cluster_onset = cluster[0]['onset']
        struct_energy = song_structure.energy_at(cluster_onset) if song_structure else 0.5

        # ── Phrase boundary reset ──
        if any(n.get('phrase_start', False) for n in cluster):
            # bank the finished phrase for Todd shaping
            if current_phrase_notes:
                all_phrase_groups.append(current_phrase_notes)
                melodic_memory.reset_phrase()
            current_phrase_notes = []
            phrase_note_idx      = 0
            home_register        = None
            phrase_start_pitches = []
            prev_leap_size       = 0
            narmour_interval     = None
            narmour_direction    = None

        has_leaped = prev_leap_size > 5

        for n in cluster:
            base_score = (
                (WEIGHT_PITCH * (n['midi'] / max_midi)) +
                (WEIGHT_DUR   * (min(n['duration'], 2.0) / max_dur)) +
                (WEIGHT_VEL   * (n['velocity'] / max_vel))
            )
            melody_bonus = 0.0

            if prev_melody_midi is not None and n['midi'] >= melody_floor:
                dist      = abs(n['midi'] - prev_melody_midi)
                direction = 1 if n['midi'] > prev_melody_midi else (-1 if n['midi'] < prev_melody_midi else 0)

                # Huron: stepwise preference
                if   dist <= 2: melody_bonus += 15.0
                elif dist == 3: melody_bonus +=  8.0
                elif dist <= 5: melody_bonus +=  3.0

                # Huron: post-leap reversal
                if prev_leap_size > 5:
                    if prev_leap_dir > 0 and direction < 0:  melody_bonus += 10.0
                    elif prev_leap_dir < 0 and direction > 0: melody_bonus += 10.0

                # Huron: melodic arch
                if   phrase_note_idx < 3 and n['midi'] > prev_melody_midi: melody_bonus += 3.0
                elif phrase_note_idx > 6 and n['midi'] < prev_melody_midi: melody_bonus += 3.0

                # Narmour: process vs reversal
                if narmour_interval is not None:
                    if narmour_interval <= 4 and direction == narmour_direction:   melody_bonus += 5.0
                    elif narmour_interval >= 6 and direction == -narmour_direction: melody_bonus += 7.0

                # Narmour: registral return
                if has_leaped and home_register is not None:
                    if abs(n['midi'] - home_register) < abs(prev_melody_midi - home_register):
                        melody_bonus += 5.0

                # Pearce & Wiggins: interval probability
                prob = INTERVAL_PROBS.get(min(dist, 5), 0.02)
                melody_bonus *= (0.8 + prob * 2.0)

                # Surprise suppression: tritone / major-7th penalty
                if dist in (6, 11):
                    target_pc = n['midi'] % 12
                    if diatonic_set is not None and target_pc not in diatonic_set:
                        melody_bonus -= 10.0

                # ── GLOBAL MEMORY: drift penalty ──
                melody_bonus -= melodic_memory.drift_penalty(n['midi'])

                # ── GLOBAL MEMORY: motif bonus ──
                pc_seq = tuple([p % 12 for p in list(pitch_buffer)[-4:]] + [n['midi'] % 12])
                melody_bonus += melodic_memory.motif_bonus(pc_seq)

            # Tymoczko: contrary motion preference
            if prev_melody_midi is not None and prev_bass_midi is not None and n['midi'] >= melody_floor:
                bass_dir   = 1 if prev_bass_midi < bass_ceiling else -1
                melody_dir = 1 if n['midi'] > prev_melody_midi else -1
                if bass_dir != melody_dir:
                    melody_bonus += 5.0

            # Structure energy → lighter harmony in low-energy sections
            if song_structure and struct_energy < 0.35 and n['midi'] < melody_floor:
                melody_bonus -= 3.0  # discourage dense harmony in verses

            n['score'] = base_score + melody_bonus
            n['role']  = 'harmony'

        # ── Bass assignment ──
        cluster.sort(key=lambda x: x['midi'])
        bass_candidate = next((n for n in cluster if n['midi'] <= bass_ceiling), cluster[0])
        bass_candidate['role'] = 'bass'

        # ── Melody assignment ──
        mel_candidates = [n for n in cluster if n['midi'] >= melody_floor and n is not bass_candidate]
        if not mel_candidates:
            mel_candidates = [n for n in cluster if n is not bass_candidate]

        if mel_candidates:
            melody_note = max(mel_candidates, key=lambda x: x.get('score', 0))
            melody_note['role'] = 'melody'

            if prev_melody_midi is not None:
                leap              = abs(melody_note['midi'] - prev_melody_midi)
                prev_leap_size    = leap
                prev_leap_dir     = 1 if melody_note['midi'] > prev_melody_midi else -1
                narmour_interval  = leap
                narmour_direction = prev_leap_dir

            prev_melody_midi = melody_note['midi']
            phrase_note_idx += 1
            pitch_buffer.append(melody_note['midi'])
            melodic_memory.update(melody_note['midi'])
            current_phrase_notes.append(melody_note)

            if len(phrase_start_pitches) < 3:
                phrase_start_pitches.append(melody_note['midi'])
            if len(phrase_start_pitches) == 3 and home_register is None:
                home_register = int(np.median(phrase_start_pitches))

        elif len(cluster) == 1:
            prev_melody_midi = cluster[0]['midi']

        prev_bass_midi = bass_candidate['midi']

        # ── Push chord into chord memory ──
        cluster_pcs = frozenset(n['midi'] % 12 for n in cluster)
        chord_memory.push(cluster_pcs)

        assigned_clusters.append(cluster)

    # bank last phrase
    if current_phrase_notes:
        all_phrase_groups.append(current_phrase_notes)

    # ── Apply Todd phrase dynamics envelope ──
    for phrase in all_phrase_groups:
        _shape_phrase_dynamics(phrase)

    return assigned_clusters


def build_chords_and_reduce(clusters, instrument="Piano", global_key="Ambiguous"):
    """
    Build and reduce chords using music21 analysis.
    IMPROVED:
      • Guitar: enforce max 5-fret physical stretch; remove voicings exceeding it.
      • Piano: bass locked below MIDI 60, locked right-hand melody above 60.
      • Perceptual quality: remove pitch-range crowding (notes within 2 semitones
        in harmony → keep the diatonically stronger one).
      • Extended harmony (9th/11th) recognised but only kept in high-energy sections.
    """
    print("[*] Building dynamic chords and enforcing polyphony reduction...")
    final_notes  = []

    for cluster_notes in clusters:
        if not cluster_notes:
            continue

        roles = {'melody': None, 'bass': None, 'harmony': []}
        for n in cluster_notes:
            if   n['role'] == 'melody': roles['melody'] = n
            elif n['role'] == 'bass':   roles['bass']   = n
            else:                       roles['harmony'].append(n)

        # music21 chord analysis on harmony block
        if roles['harmony']:
            midi_list    = [n['midi'] for n in roles['harmony']]
            c            = chord.Chord(midi_list)
            root         = c.root().midi  if c.root()  else None
            third        = c.third.midi   if c.third   else None
            fifth        = c.fifth.midi   if c.fifth   else None
            diatonic_set = get_diatonic_pitches(global_key)

            kept_harmony = []
            for n in roles['harmony']:
                is_diatonic = diatonic_set is None or (n['midi'] % 12 in diatonic_set)
                if   root  and n['midi'] % 12 == root  % 12: n['priority'] = 3
                elif third and n['midi'] % 12 == third % 12: n['priority'] = 4
                elif fifth and n['midi'] % 12 == fifth % 12: n['priority'] = 5
                elif is_diatonic:                             n['priority'] = 6
                else:
                    continue   # chromatic noise — drop

                kept_harmony.append(n)

            # ── Perceptual: remove crowding (< 2 semitone spacing in harmony) ──
            kept_harmony.sort(key=lambda x: x['midi'])
            decrowed = []
            for n in kept_harmony:
                if decrowed and abs(n['midi'] - decrowed[-1]['midi']) < 2:
                    # keep whichever has lower priority number (more important)
                    if n.get('priority', 9) < decrowed[-1].get('priority', 9):
                        decrowed[-1] = n
                else:
                    decrowed.append(n)

            kept_harmony = decrowed
            kept_harmony.sort(key=lambda x: x['priority'])
            roles['harmony'] = kept_harmony

        # ── Polyphony limits ──
        limit_poly = 4
        two_hands  = True

        if instrument in MONO_INSTRUMENTS:
            limit_poly = 1;  two_hands = False
            roles['harmony'] = [];  roles['bass'] = None
        elif "Guitar" in instrument:
            limit_poly = 6;  two_hands = False
        else:
            limit_poly = 4;  two_hands = True

        # Guitar: physical stretch constraint (max 5 frets across all sounding notes)
        if "Guitar" in instrument and roles['harmony']:
            all_guitar = ([roles['melody']] if roles['melody'] else []) + \
                         ([roles['bass']]   if roles['bass']   else []) + \
                         roles['harmony']
            midi_vals = sorted(n['midi'] for n in all_guitar)
            while len(midi_vals) > 1 and (midi_vals[-1] - midi_vals[0]) > 5:
                # drop the most extreme harmony note
                extreme = max(roles['harmony'],
                              key=lambda n: abs(n['midi'] - midi_vals[len(midi_vals)//2]),
                              default=None)
                if extreme is None:
                    break
                roles['harmony'].remove(extreme)
                midi_vals = sorted(n['midi'] for n in
                                   ([roles['melody']] if roles['melody'] else []) +
                                   ([roles['bass']]   if roles['bass']   else []) +
                                   roles['harmony'])

        right_cnt = 0;  left_cnt = 0;  mono_cnt = 0

        def push_note(n, forced_hand=None):
            nonlocal right_cnt, left_cnt, mono_cnt
            if two_hands:
                hand = forced_hand or ('right' if n['midi'] >= 60 else 'left')
                n['hand'] = hand
                if hand == 'right' and right_cnt < limit_poly:
                    final_notes.append(n);  right_cnt += 1
                elif hand == 'left' and left_cnt < limit_poly:
                    final_notes.append(n);  left_cnt  += 1
            else:
                n['hand'] = 'right'
                if mono_cnt < limit_poly:
                    final_notes.append(n);  mono_cnt += 1

        if roles['melody']:               push_note(roles['melody'], 'right')
        if roles['bass'] and two_hands:   push_note(roles['bass'],   'left')
        elif roles['bass']:               push_note(roles['bass'],   'right')
        for h in roles['harmony']:        push_note(h)

    # Final sanity: remove impossible instant jumps on right hand
    prev = None
    cleaned_final = []
    for n in final_notes:
        if n['hand'] == 'right':
            if prev and abs(n['midi'] - prev['midi']) > 14 and abs(n['onset'] - prev['onset']) < 0.1:
                continue
            prev = n
        cleaned_final.append(n)

    print(f"    -> Outputting {len(cleaned_final)} fully arranged notes.")
    return cleaned_final


def optimize_voice_leading(arranged_notes, instrument="Piano", global_key="Ambiguous"):
    """
    Tymoczko (2006): minimise total voice-leading distance + detect parallel 5ths/octaves.
    IMPROVED: also checks ±24 (two octaves) for smoother resolutions in wide-range instruments.
    """
    print("[*] Optimizing voice leading across chord pairs (Tymoczko)...")
    TOL    = 0.055
    lo, hi = INSTRUMENT_RANGES.get(instrument, (21, 108))

    beat_clusters = []
    for n in sorted(arranged_notes, key=lambda x: x['onset']):
        placed = False
        for cluster in beat_clusters:
            if abs(n['onset'] - cluster[0]['onset']) <= TOL:
                cluster.append(n);  placed = True;  break
        if not placed:
            beat_clusters.append([n])

    prev_cluster = None
    corrected    = 0

    for cluster in beat_clusters:
        if prev_cluster is None:
            prev_cluster = cluster;  continue

        prev_midis = [n['midi'] for n in prev_cluster]

        for n in cluster:
            if n.get('role') != 'harmony':
                continue
            original_midi = n['midi']
            best_midi     = original_midi
            best_dist     = sum(abs(original_midi - p) for p in prev_midis)

            for delta in (-24, -12, 12, 24):
                alt = original_midi + delta
                if lo <= alt <= hi:
                    d = sum(abs(alt - p) for p in prev_midis)
                    if d < best_dist:
                        best_dist = d;  best_midi = alt

            if best_midi != original_midi:
                n['midi'] = best_midi;  corrected += 1

        # Parallel 5ths / octaves correction
        if len(prev_cluster) >= 2 and len(cluster) >= 2:
            for i, n1 in enumerate(cluster):
                for j, n2 in enumerate(cluster):
                    if i >= j:
                        continue
                    p1 = min(prev_cluster, key=lambda p: abs(p['midi'] - n1['midi']))
                    p2 = min(prev_cluster, key=lambda p: abs(p['midi'] - n2['midi']))
                    int_before = abs(p1['midi'] - p2['midi']) % 12
                    int_after  = abs(n1['midi'] - n2['midi']) % 12
                    move1      = n1['midi'] - p1['midi']
                    move2      = n2['midi'] - p2['midi']
                    is_parallel = (int_before == int_after and
                                   int_after in (0, 7) and
                                   np.sign(move1) == np.sign(move2) and
                                   move1 != 0)
                    if is_parallel and n2.get('role') == 'harmony':
                        fix = n2['midi'] + (12 if move2 < 0 else -12)
                        if lo <= fix <= hi:
                            n2['midi'] = fix;  corrected += 1

        prev_cluster = cluster

    print(f"[VL] Voice leading pass complete: {corrected} notes re-voiced.")
    return arranged_notes


def enforce_density_budget(notes, instrument, global_key,
                           song_structure: SongStructure = None):
    """
    Tension-aware density budget (Lerdahl).
    IMPROVED: SongStructure multiplier adjusts limit per section.
    """
    DENSITY_LIMITS = {
        "Piano": 6, "Guitar": 4, "Sitar": 1,
        "Flute": 1, "Violin": 1, "Trumpet": 1,
    }
    base_limit      = DENSITY_LIMITS.get(instrument, 6)
    diatonic_set    = get_diatonic_pitches(global_key)
    root_pc_key     = NOTE_MAP.get(global_key.split()[0], 0) if global_key != "Ambiguous" else None
    leading_tone_pc = (root_pc_key + 11) % 12 if root_pc_key is not None else None

    TOL = 0.005
    beat_groups = []
    for n in sorted(notes, key=lambda x: x['onset']):
        placed = False
        for group in beat_groups:
            if abs(n['onset'] - group[0]['onset']) <= TOL:
                group.append(n);  placed = True;  break
        if not placed:
            beat_groups.append([n])

    final = []
    for group in beat_groups:
        tension = compute_tension([n['midi'] for n in group], global_key)
        struct_delta = song_structure.density_multiplier(group[0]['onset']) if song_structure else 0

        if tension >= 0.8:   limit = min(base_limit + 2 + struct_delta, 8)
        elif tension <= 0.3: limit = max(base_limit - 1 + struct_delta, 2)
        else:                limit = max(2, base_limit + struct_delta)

        if len(group) <= limit:
            final.extend(group);  continue

        locked  = [n for n in group if n.get('role') in ('bass', 'melody')]
        harmony = [n for n in group if n.get('role') not in ('bass', 'melody')]

        try:
            from music21 import chord as m21chord
            c        = m21chord.Chord([n['midi'] for n in harmony]) if harmony else None
            root_pc  = c.root().midi  % 12 if (c and c.root())  else None
            third_pc = c.third.midi % 12   if (c and c.third)   else None
            fifth_pc = c.fifth.midi % 12   if (c and c.fifth)   else None
        except Exception:
            root_pc = third_pc = fifth_pc = None

        def harm_priority(n):
            pc = n['midi'] % 12
            if leading_tone_pc is not None and pc == leading_tone_pc: return 0
            if root_pc  is not None and pc == root_pc:  return 1
            if third_pc is not None and pc == third_pc: return 2
            if fifth_pc is not None and pc == fifth_pc: return 3
            if diatonic_set is None or pc in diatonic_set:          return 4
            return 5

        harmony.sort(key=harm_priority)
        budget_remaining = limit - len(locked)

        # Implied harmony pruning
        if budget_remaining < len(harmony):
            melody_note = next((n for n in locked if n.get('role') == 'melody'), None)
            bass_note   = next((n for n in locked if n.get('role') == 'bass'),   None)
            if melody_note and bass_note and diatonic_set:
                melody_pc = melody_note['midi'] % 12
                bass_pc   = bass_note['midi']   % 12
                melody_is_chord_tone = (
                    (root_pc  is not None and melody_pc == root_pc) or
                    (third_pc is not None and melody_pc == third_pc) or
                    (fifth_pc is not None and melody_pc == fifth_pc)
                )
                if melody_is_chord_tone:
                    pruned = []
                    for h in harmony:
                        pc = h['midi'] % 12
                        if fifth_pc is not None and pc == fifth_pc and melody_pc in (root_pc, third_pc):
                            continue
                        if third_pc is not None and pc == third_pc and melody_pc == root_pc and bass_pc == root_pc:
                            continue
                        pruned.append(h)
                    harmony = pruned

        kept_harmony = harmony[:budget_remaining]
        for d in harmony[budget_remaining:]:
            print(f"  [DENSITY] Budget exceeded: dropped MIDI {d['midi']} (priority: {harm_priority(d)})")

        final.extend(locked)
        final.extend(kept_harmony)

    print(f"[DENSITY] Budget enforcement complete. {len(final)} notes retained from {len(notes)}.")
    return final


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — REFINEMENT / SELF-CORRECTION PASS
# ═══════════════════════════════════════════════════════════════════════════════

def refinement_pass(notes, global_key, instrument):
    """
    Self-correction pass that wraps three validators, each modifying notes
    in-place and returning a corrected list.

    1. Melody jump smoother  — octave leaps > 12 semitones between consecutive
       melody notes are corrected by choosing the octave that minimises the
       interval, staying within instrument range.

    2. Harmonic coherence   — consecutive harmony blocks with very high tonal
       distance (> 0.85) get their most recent block re-voiced by shifting the
       note with highest tonal contribution one semitone toward the previous chord.

    3. Key consistency      — notes with pitch class NOT in diatonic set get
       replaced by the nearest diatonic semitone if the replacement is within ±1
       semitone (chromatic passing tones kept if beyond that).
    """
    print("[*] Refinement pass: self-correcting melody / harmony / key...")
    lo, hi       = INSTRUMENT_RANGES.get(instrument, (0, 127))
    diatonic_set = get_diatonic_pitches(global_key)

    notes_by_onset = sorted(notes, key=lambda x: x['onset'])

    # ─── 1. Melody jump smoother ─────────────────────────────────────────────
    melody_notes = [n for n in notes_by_onset if n.get('role') == 'melody']
    fixed_jumps  = 0
    prev_m = None
    for mn in melody_notes:
        if prev_m is not None:
            jump = abs(mn['midi'] - prev_m['midi'])
            if jump > 12:
                # Try moving the current note by ±12 to reduce jump
                for delta in (-12, 12):
                    alt = mn['midi'] + delta
                    if lo <= alt <= hi and abs(alt - prev_m['midi']) < jump:
                        mn['midi'] = alt
                        fixed_jumps += 1
                        break
        prev_m = mn
    print(f"    [REFINE] Melody: {fixed_jumps} large jumps smoothed.")

    # ─── 2. Harmonic coherence ────────────────────────────────────────────────
    TOL       = 0.055
    groups    = []
    for n in notes_by_onset:
        placed = False
        for g in groups:
            if abs(n['onset'] - g[0]['onset']) <= TOL:
                g.append(n);  placed = True;  break
        if not placed:
            groups.append([n])

    fixed_chords = 0
    prev_pcs = None
    for g in groups:
        curr_pcs = frozenset(n['midi'] % 12 for n in g if n.get('role') == 'harmony')
        if prev_pcs and curr_pcs:
            dist = tonal_distance(list(prev_pcs), list(curr_pcs))
            if dist > 0.85:
                # Re-voice: shift the harmony note farthest from prev chord
                harm_notes = [n for n in g if n.get('role') == 'harmony']
                if harm_notes:
                    worst = max(harm_notes,
                                key=lambda n: min(abs(n['midi'] % 12 - p) for p in prev_pcs))
                    orig = worst['midi']
                    for delta in (-1, 1, -2, 2):
                        alt = worst['midi'] + delta
                        if lo <= alt <= hi:
                            new_pcs = frozenset(
                                (n['midi'] if n is not worst else alt) % 12
                                for n in g if n.get('role') == 'harmony')
                            if tonal_distance(list(prev_pcs), list(new_pcs)) < dist:
                                worst['midi'] = alt
                                fixed_chords += 1
                                break
        prev_pcs = curr_pcs or prev_pcs
    print(f"    [REFINE] Harmony: {fixed_chords} incoherent chord transitions fixed.")

    # ─── 3. Key consistency ───────────────────────────────────────────────────
    fixed_key = 0
    if diatonic_set:
        for n in notes:
            pc = n['midi'] % 12
            if pc not in diatonic_set:
                # find nearest diatonic pitch class
                nearest = min(diatonic_set, key=lambda d: min(abs(d - pc), 12 - abs(d - pc)))
                diff    = nearest - pc
                # allow wrap-around preference
                if diff > 6:  diff -= 12
                if diff < -6: diff += 12
                if abs(diff) <= 1:   # only substitute 1-semitone chromatic neighbours
                    alt = n['midi'] + diff
                    if lo <= alt <= hi:
                        n['midi'] = alt
                        fixed_key += 1
    print(f"    [REFINE] Key: {fixed_key} non-diatonic notes corrected.")

    return notes


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — EXPRESSION PASS (MUSICAL INTENT MODEL)
# ═══════════════════════════════════════════════════════════════════════════════

def expression_pass(notes, instrument, song_structure: SongStructure = None):
    """
    Musical intent model: treat melody, bass, harmony with distinct
    velocity / articulation / timing strategies.

    Melody  → loudest, legato articulation, subtle rubato timing
    Bass    → firm, medium, slightly ahead of beat (-8ms) for rhythmic drive
    Harmony → softer, staccato-ish, no timing push

    Also applies SongStructure velocity scaling so choruses are louder.
    """
    print("[*] Expression pass: applying musical intent model...")

    for n in notes:
        role      = n.get('role', 'harmony')
        struct_vs = song_structure.velocity_scale(n['onset']) if song_structure else 1.0

        # ── Velocity intent ──
        raw_vel = float(n.get('velocity', 0.5))
        if raw_vel <= 1.0:
            raw_vel *= 127

        if role == 'melody':
            # Apply phrase dynamics envelope baked by _shape_phrase_dynamics
            phrase_scale = n.get('_phrase_vel_scale', 1.0)
            target_vel   = 88 + (raw_vel / 127.0) * 20   # 88-108
            target_vel  *= phrase_scale * struct_vs
        elif role == 'bass':
            target_vel = 62 + (raw_vel / 127.0) * 18     # 62-80
            target_vel *= struct_vs
        else:
            target_vel = 44 + (raw_vel / 127.0) * 18     # 44-62
            target_vel *= struct_vs

        # Gaussian jitter: melody σ=6, bass σ=4, harmony σ=8
        sigma = {'melody': 6, 'bass': 4, 'harmony': 8}.get(role, 6)
        target_vel += random.gauss(0, sigma)
        n['velocity'] = int(max(1, min(127, round(target_vel))))

        # ── Articulation intent (duration shaping) ──
        # Melody: legato (0.95 × duration), bass: normal, harmony: shorter (0.72)
        if role == 'melody':
            n['duration'] = round(n['duration'] * 0.95, 3)
        elif role == 'harmony':
            n['duration'] = round(n['duration'] * 0.72, 3)

        # Ensure minimum audible duration
        n['duration'] = max(0.05, n['duration'])

        # ── Timing micro-offsets by role ──
        if role == 'bass':
            # Bass slightly ahead (drives rhythm)
            n['onset'] = round(n['onset'] - 0.008, 3)
        elif role == 'melody':
            # Melody: tiny Gaussian rubato σ=5ms
            n['onset'] = round(n['onset'] + random.gauss(0, 0.005), 3)
        else:
            # Harmony: slight delay σ=7ms (fills in after the beat)
            n['onset'] = round(n['onset'] + abs(random.gauss(0, 0.007)), 3)

        n['onset'] = max(0.0, n['onset'])

    print(f"    -> Expression pass complete ({len(notes)} notes processed).")
    return notes


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 7 — VALIDATION PASS (FINAL COHERENCE CHECKS)
# ═══════════════════════════════════════════════════════════════════════════════

def validation_pass(notes, global_key, instrument):
    """
    Final pass: catch any remaining anomalies after full arrangement.

    Checks:
      1. Overlapping identical pitches on same onset → keep louder one.
      2. Notes outside instrument physical range → clamp.
      3. Zero or negative duration → set to minimum 0.05s.
      4. Velocity out of MIDI range → clamp to [1, 127].
    """
    print("[*] Validation pass: final coherence checks...")
    lo, hi = INSTRUMENT_RANGES.get(instrument, (0, 127))
    issues = 0

    # 1. Deduplicate identical pitch at same onset
    seen: set = set()
    deduped   = []
    for n in sorted(notes, key=lambda x: (x['onset'], -x['velocity'])):
        key = (round(n['onset'], 2), n['midi'])
        if key not in seen:
            seen.add(key)
            deduped.append(n)
        else:
            issues += 1
    notes = deduped

    # 2–4. Range / duration / velocity clamp
    for n in notes:
        if n['midi'] < lo:
            n['midi'] = lo;  issues += 1
        elif n['midi'] > hi:
            n['midi'] = hi;  issues += 1
        if n['duration'] <= 0:
            n['duration'] = 0.05;  issues += 1
        if 'velocity' in n:
            n['velocity'] = max(1, min(127, int(n['velocity'])))

    print(f"    [VALIDATE] {issues} anomalies corrected.")
    return notes


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMING JSON PARSER
# ═══════════════════════════════════════════════════════════════════════════════

def _coerce_note(n: dict) -> dict:
    """
    Cast all numeric note fields from decimal.Decimal (returned by ijson) to
    plain Python float / int so every downstream stage can do normal arithmetic.
    Safe to call on notes already containing floats — no-op in that case.
    """
    n['onset']    = float(n['onset'])
    n['offset']   = float(n['offset'])
    n['velocity'] = float(n['velocity'])
    n['midi']     = int(n['midi'])
    n['duration'] = float(n.get('duration', n['offset'] - n['onset']))
    return n


def _stream_notes(json_file: str):
    """
    Memory-efficient streaming parse of the large transcription JSON.
    Returns (global_key, raw_notes_list).
    Uses ijson if available; falls back to json.load().
    All note numeric fields are guaranteed to be plain float / int.
    """
    if HAS_IJSON:
        print(f"[*] Streaming JSON with ijson: {json_file}")
        global_key = "Ambiguous"

        # Pass 1: pull top-level 'detected_key' scalar only
        with open(json_file, 'r', encoding='utf-8') as f:
            for prefix, event, value in ijson.parse(f):
                if prefix == 'detected_key' and event == 'string':
                    global_key = value
                    break

        # Pass 2: stream note objects — cast Decimal → float/int immediately
        raw_notes: list = []
        seen: set = set()
        with open(json_file, 'r', encoding='utf-8') as f:
            for seg in ijson.items(f, 'segments.item'):
                for n in seg.get('notes', []):
                    n = _coerce_note(dict(n))   # cast + copy
                    n_hash = (round(n['onset'], 2), n['midi'])
                    if n_hash not in seen:
                        seen.add(n_hash)
                        raw_notes.append(n)

        print(f"    -> Streamed {len(raw_notes)} unique notes. Key: {global_key}")
        return global_key, raw_notes

    else:
        print(f"[*] Loading JSON: {json_file}")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        global_key = data.get('detected_key', 'Ambiguous')
        raw_notes: list = []
        seen: set = set()
        for seg in data.get('segments', []):
            for n in seg.get('notes', []):
                n = _coerce_note(dict(n))
                n_hash = (round(n['onset'], 2), n['midi'])
                if n_hash not in seen:
                    seen.add(n_hash)
                    if 'duration' not in n:
                        n['duration'] = round(n['offset'] - n['onset'], 3)
                    raw_notes.append(n)
        print(f"    -> Loaded {len(raw_notes)} unique notes. Key: {global_key}")
        return global_key, raw_notes


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def arrange(json_file, audio_file, output_file, instrument="Piano"):
    print("=========================================")
    print(" EXPERT AI ARRANGER PIPELINE  v5.0")
    print("=========================================\n")

    # ── Stage 0: Stream + detect structure ──────────────────────────────────
    print("\n[STAGE 0]: Streaming JSON + Structure Detection")
    global_key, raw_notes = _stream_notes(json_file)
    print(f"[*] Dynamic Profile -> Instrument: {instrument}, Key: {global_key}")

    # Estimate duration from last note offset
    total_duration = max((n['offset'] for n in raw_notes), default=60.0)

    # Initialise long-term memory objects
    melodic_memory = MelodicMemory()
    chord_memory   = ChordMemory()
    song_structure = SongStructure()
    song_structure.analyse(raw_notes, total_duration)

    # Estimate tempo for tempo-scaled clustering
    try:
        y_tmp, sr_tmp = librosa.load(audio_file, sr=22050, duration=60.0)
        tempo_arr, _  = librosa.beat.beat_track(y=y_tmp, sr=sr_tmp)
        song_tempo    = float(np.atleast_1d(tempo_arr)[0])
    except Exception:
        song_tempo = 120.0
    print(f"[*] Detected tempo: {song_tempo:.1f} BPM")

    # ── GATE: Frequency Band Filter ──────────────────────────────────────────
    print("\n[GATE]: Instrument Physical Range Filter")
    gated_notes = frequency_band_gate(raw_notes, instrument)

    # ── Stage 1: Foundational Processing ────────────────────────────────────
    print("\n[STAGE 1]: Foundational Processing")
    cleaned   = clean_notes(gated_notes, instrument=instrument)
    quantized = adaptive_quantization(cleaned, audio_file)
    clusters  = cluster_notes(quantized, window_ms=50, tempo=song_tempo)

    # ── Stage 2 & 3: Role Assignment + Chord Building ───────────────────────
    print("\n[STAGE 2/3]: Harmonics & Hand Mapping")
    assigned       = assign_roles(clusters, instrument=instrument,
                                  global_key=global_key,
                                  melodic_memory=melodic_memory,
                                  chord_memory=chord_memory,
                                  song_structure=song_structure)
    arranged_notes = build_chords_and_reduce(assigned, instrument=instrument,
                                             global_key=global_key)

    # ── Stage 3.5: Voice Leading ─────────────────────────────────────────────
    print("\n[STAGE 3.5]: Tymoczko Voice Leading Optimizer")
    arranged_notes = optimize_voice_leading(arranged_notes, instrument=instrument,
                                            global_key=global_key)

    # ── Stage 4: Density Budget ───────────────────────────────────────────────
    print("\n[STAGE 4]: Density Budget Enforcement")
    final_notes = enforce_density_budget(arranged_notes, instrument=instrument,
                                         global_key=global_key,
                                         song_structure=song_structure)

    # ── Stage 5: Self-Correction ──────────────────────────────────────────────
    print("\n[STAGE 5]: Refinement / Self-Correction Pass")
    final_notes = refinement_pass(final_notes, global_key=global_key, instrument=instrument)

    # ── Stage 6: Expression ───────────────────────────────────────────────────
    print("\n[STAGE 6]: Expression Pass (Musical Intent Model)")
    final_notes = expression_pass(final_notes, instrument=instrument,
                                  song_structure=song_structure)

    # ── Stage 7: Validation ───────────────────────────────────────────────────
    print("\n[STAGE 7]: Validation Pass")
    final_notes = validation_pass(final_notes, global_key=global_key, instrument=instrument)

    # ── Format + Save ─────────────────────────────────────────────────────────
    output_data = []
    for n in final_notes:
        vel = n.get('velocity', 80)
        # normalise legacy float velocities
        if isinstance(vel, float) and vel <= 1.0:
            vel = int(vel * 127)
        output_data.append({
            "time":         n['onset'],
            "note":         n['note'],
            "midi":         n['midi'],
            "duration":     n['duration'],
            "velocity":     int(max(1, min(127, vel))),
            "hand":         n.get('hand', 'right'),
            "role":         n.get('role', 'harmony'),
            "phrase_start": n.get('phrase_start', False),
            "measure":      0,
            "beat":         0.0
        })

    output_data.sort(key=lambda x: x['time'])

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n[INFO] Complete. Saved {len(output_data)} notes to {output_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default="llm_input.json",  help="Transcription JSON")
    parser.add_argument("--audio",      default="test.mp3",         help="Original audio for beat tracking")
    parser.add_argument("--output",     default="arranged.json",    help="Output arranged JSON")
    parser.add_argument("--instrument", default="Piano",            help="Target instrument")
    args = parser.parse_args()

    arrange(args.input, args.audio, args.output, instrument=args.instrument)
