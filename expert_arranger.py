import json
import argparse
import numpy as np
import librosa
from collections import defaultdict
from music21 import chord, note, pitch

NOTE_MAP = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
            "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
            "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}

def get_diatonic_pitches(key_str):
    if key_str == "Ambiguous": return None
    parts = key_str.split(" ")
    if len(parts) != 2: return None
    root_str, scale_type = parts
    root_midi = NOTE_MAP.get(root_str, 0)
    
    if scale_type == "Major":
        intervals = [0, 2, 4, 5, 7, 9, 11]
    else: # Minor
        intervals = [0, 2, 3, 5, 7, 8, 10]
        
    return set([(root_midi + iv) % 12 for iv in intervals])


# ─── MUSIC THEORY CONSTANTS ──────────────────────────────────────────────────

# Pearce & Wiggins (2006): Empirical melodic interval probability distribution
INTERVAL_PROBS = {0: 0.16, 1: 0.14, 2: 0.20, 3: 0.10, 4: 0.08, 5: 0.07}

def compute_tension(midi_list, key_str):
    """
    Lerdahl (2001): Tonal Pitch Space tension score [0-1] for a chord cluster.
    High = dissonant (diminished, dominant 7th, leading tone).
    Low  = consonant (tonic, subdominant).
    """
    if not midi_list or key_str == "Ambiguous":
        return 0.5
    diatonic = get_diatonic_pitches(key_str)
    root_pc  = NOTE_MAP.get(key_str.split()[0], 0)
    pcs = [m % 12 for m in midi_list]

    # Detect structural tension indicators
    has_tritone      = any(abs(pcs[i] - pcs[j]) % 12 == 6
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


# ─── INSTRUMENT PHYSICAL RANGES ─────────────────────────────────────────────

INSTRUMENT_RANGES = {
    "Piano":   (21, 108),
    "Guitar":  (40, 84),
    "Sitar":   (50, 77),
    "Flute":   (60, 96),
    "Violin":  (55, 103),
    "Trumpet": (58, 94),
}

def frequency_band_gate(notes, instrument):
    """
    Permanently delete notes falling outside the instrument's physical MIDI range.
    """
    lo, hi = INSTRUMENT_RANGES.get(instrument, (0, 127))
    kept = []
    deleted = 0
    for n in notes:
        if n['midi'] < lo:
            print(f"  [GATE] Deleted MIDI {n['midi']} ({n['note']}): below {instrument} min ({lo})")
            deleted += 1
        elif n['midi'] > hi:
            print(f"  [GATE] Deleted MIDI {n['midi']} ({n['note']}): above {instrument} max ({hi})")
            deleted += 1
        else:
            kept.append(n)
    print(f"[GATE] Frequency band gate: removed {deleted} out-of-range notes, kept {len(kept)}.")
    return kept

# ─── CORE PIPELINE STAGES ───────────────────────────────────────────────

def clean_notes(notes, instrument="Piano"):
    """
    Remove extreme short noises and merge directly overlapping notes of same pitch.
    """
    if not notes:
        return []
        
    print(f"[*] Cleaning {len(notes)} raw notes...")
    valid_notes = [n for n in notes if n['offset'] - n['onset'] >= 0.1]
    
    # Sort by pitch, then onset
    valid_notes.sort(key=lambda x: (x['midi'], x['onset']))
    
    is_mono = instrument in ["Sitar", "Flute", "Violin", "Trumpet"]
    gap_limit = 0.35 if is_mono else 0.15
    
    merged_notes = []
    current_note = None
    
    for note in valid_notes:
        if current_note is None:
            current_note = dict(note)
            current_note['phrase_start'] = True
            continue
            
        # Rhythmic Phrasing: Rest Detection
        if note['onset'] - current_note['offset'] > 0.5:
            merged_notes.append(current_note)
            current_note = dict(note)
            current_note['phrase_start'] = True # Mark as start of new phrase
            continue
            
        # Instrument-Aware Legato Gap-Tying
        if note['midi'] == current_note['midi'] and note['onset'] <= current_note['offset'] + gap_limit:
            # Merge offsets and take the max velocity
            current_note['offset'] = max(current_note['offset'], note['offset'])
            current_note['velocity'] = max(current_note['velocity'], note['velocity'])
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
    Snap note boundaries using dynamic local beat tracking instead of global rigid tempo.
    """
    print(f"[*] Running adaptive quantization on audio: {audio_path}")
    
    # Load audio and extract beat frames natively
    y, sr = librosa.load(audio_path, sr=22050)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    
    # We will compute 16th note subdivisions based on the dynamic beats
    # For now, as Stage 1 implementation, we snap to nearest beat division (1/4th of a beat)
    
    def find_nearest_grid(time_val, beats_array, subdivision=4):
        # Very basic grid snapping. In advanced stages, interpolate between beats.
        # Handle cases where time is before first beat or after last beat
        if len(beats_array) == 0:
            return time_val
            
        if time_val <= beats_array[0]:
            beat_duration = beats_array[1] - beats_array[0] if len(beats_array)>1 else 0.5
            return beats_array[0] - (beat_duration) * round((beats_array[0]-time_val)/beat_duration * subdivision) / subdivision
            
        idx = np.searchsorted(beats_array, time_val)
        if idx >= len(beats_array):
            beat_duration = beats_array[-1] - beats_array[-2]
            dist = time_val - beats_array[-1]
            return beats_array[-1] + beat_duration * round(dist/beat_duration * subdivision) / subdivision
            
        t0, t1 = beats_array[idx-1], beats_array[idx]
        grid_step = (t1 - t0) / subdivision
        
        # Determine how many grid steps we are past t0
        steps = round((time_val - t0) / grid_step)
        return t0 + (steps * grid_step)
        
    quantized = []
    max_vel = max([n['velocity'] for n in notes]) if notes else 1.0
    
    for n in notes:
        # Clamp bottom 10th percentile
        clamped_vel = max(n['velocity'], max_vel * 0.1)
        snap_strength = min(1.0, (clamped_vel / max_vel) * 1.5)
        
        q_onset_grid = find_nearest_grid(n['onset'], beat_times)
        q_offset_grid = find_nearest_grid(n['offset'], beat_times)
        
        q_onset = n['onset'] + (q_onset_grid - n['onset']) * snap_strength
        q_offset = n['offset'] + (q_offset_grid - n['offset']) * snap_strength
        
        # Ensure it didn't snap to length zero
        if q_offset <= q_onset:
            q_offset = q_onset + ((beat_times[1]-beat_times[0]) / 4 if len(beat_times)>1 else 0.1)
            
        quantized.append({
            **n,
            'onset': round(q_onset, 3),
            'offset': round(q_offset, 3),
            'duration': round(q_offset - q_onset, 3)
        })
        
    # Sort back by original time logical flow
    quantized.sort(key=lambda x: x['onset'])
    return quantized

def cluster_notes(notes, window_ms=50):
    """
    Group events falling in ~50ms windows to form valid harmonic clusters.
    """
    print(f"[*] Clustering notes within {window_ms}ms windows...")
    clusters = []
    if not notes:
        return clusters
        
    current_cluster = [notes[0]]
    cluster_time = notes[0]['onset']
    
    threshold = window_ms / 1000.0
    
    for note in notes[1:]:
        if note['onset'] - cluster_time <= threshold:
            current_cluster.append(note)
        else:
            clusters.append(current_cluster)
            current_cluster = [note]
            cluster_time = note['onset']
            
    if current_cluster:
        clusters.append(current_cluster)
        
    print(f"    -> Assembled {len(clusters)} time-aligned clusters.")
    return clusters

def assign_roles(clusters, instrument="Piano", global_key="Ambiguous"):
    """
    V4 Music Theory Engine — integrates:
    Huron (2006): stepwise preference, post-leap reversal, melodic arch
    Narmour (1990): process/reversal two-state machine, registral return
    Pearce & Wiggins (2006): interval probability weighting
    Tymoczko (2006): contrary motion preference
    Surprise suppression: tritone/major-7th penalty
    """
    print("[*] Assigning musical roles (V4 Music Theory Engine)...")

    lo, hi    = INSTRUMENT_RANGES.get(instrument, (0, 127))
    span      = hi - lo
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

    # ── Cross-cluster memory ──
    prev_melody_midi  = None
    prev_bass_midi    = None
    prev_leap_size    = 0
    prev_leap_dir     = 0      # +1 up, -1 down
    narmour_interval  = None
    narmour_direction = None

    # ── Phrase-level memory ──
    phrase_note_idx      = 0
    home_register        = None
    phrase_start_pitches = []

    assigned_clusters = []

    for cluster in clusters:
        if not cluster:
            continue

        # Reset phrase state on boundary
        if any(n.get('phrase_start', False) for n in cluster):
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

                # ── Huron: Refined stepwise preference ──
                if   dist <= 2: melody_bonus += 15.0
                elif dist == 3: melody_bonus +=  8.0
                elif dist <= 5: melody_bonus +=  3.0

                # ── Huron: Post-leap reversal ──
                if prev_leap_size > 5:
                    if prev_leap_dir > 0 and direction < 0: melody_bonus += 10.0
                    elif prev_leap_dir < 0 and direction > 0: melody_bonus += 10.0

                # ── Huron: Melodic arch shaping ──
                if phrase_note_idx < 3 and n['midi'] > prev_melody_midi:  melody_bonus += 3.0
                elif phrase_note_idx > 6 and n['midi'] < prev_melody_midi: melody_bonus += 3.0

                # ── Narmour: Process vs Reversal ──
                if narmour_interval is not None:
                    if narmour_interval <= 4 and direction == narmour_direction:   melody_bonus += 5.0
                    elif narmour_interval >= 6 and direction == -narmour_direction: melody_bonus += 7.0

                # ── Narmour: Registral return pull ──
                if has_leaped and home_register is not None:
                    if abs(n['midi'] - home_register) < abs(prev_melody_midi - home_register):
                        melody_bonus += 5.0

                # ── Pearce & Wiggins: Interval probability weighting ──
                prob = INTERVAL_PROBS.get(min(dist, 5), 0.02)
                melody_bonus *= (0.8 + prob * 2.0)

                # ── Surprise suppression: tritone/major-7th penalty ──
                if dist in (6, 11):
                    target_pc = n['midi'] % 12
                    if diatonic_set is not None and target_pc not in diatonic_set:
                        melody_bonus -= 10.0

            # ── Tymoczko: Contrary motion preference ──
            if prev_melody_midi is not None and prev_bass_midi is not None and n['midi'] >= melody_floor:
                bass_dir   = 1 if prev_bass_midi < bass_ceiling else -1  # simplified bass direction proxy
                melody_dir = 1 if n['midi'] > prev_melody_midi else -1
                if bass_dir != melody_dir:
                    melody_bonus += 5.0

            n['score'] = base_score + melody_bonus
            n['role']  = 'harmony'

        # ── Bass assignment  (zone-constrained) ──
        cluster.sort(key=lambda x: x['midi'])
        bass_candidate = next((n for n in cluster if n['midi'] <= bass_ceiling), cluster[0])
        bass_candidate['role'] = 'bass'

        # ── Melody assignment (zone-constrained, highest score) ──
        mel_candidates = [n for n in cluster if n['midi'] >= melody_floor and n is not bass_candidate]
        if not mel_candidates:
            mel_candidates = [n for n in cluster if n is not bass_candidate]

        if mel_candidates:
            melody_note = max(mel_candidates, key=lambda x: x.get('score', 0))
            melody_note['role'] = 'melody'

            if prev_melody_midi is not None:
                leap               = abs(melody_note['midi'] - prev_melody_midi)
                prev_leap_size     = leap
                prev_leap_dir      = 1 if melody_note['midi'] > prev_melody_midi else -1
                narmour_interval   = leap
                narmour_direction  = prev_leap_dir

            prev_melody_midi = melody_note['midi']
            phrase_note_idx += 1

            if len(phrase_start_pitches) < 3:
                phrase_start_pitches.append(melody_note['midi'])
            if len(phrase_start_pitches) == 3 and home_register is None:
                home_register = int(np.median(phrase_start_pitches))
        elif len(cluster) == 1:
            prev_melody_midi = cluster[0]['midi']

        prev_bass_midi = bass_candidate['midi']
        assigned_clusters.append(cluster)

    return assigned_clusters

def build_chords_and_reduce(clusters, instrument="Piano", global_key="Ambiguous"):
    """
    Utilize music21 to process inner harmony, simplify chords, and strict hierarchical dropout.
    """
    print("[*] Building dynamic chords and enforcing polyphony reduction...")
    final_notes = []
    
    for cluster_notes in clusters:
        if not cluster_notes: continue
        
        roles = { 'melody': None, 'bass': None, 'harmony': [] }
        for n in cluster_notes:
            if n['role'] == 'melody': roles['melody'] = n
            elif n['role'] == 'bass': roles['bass'] = n
            else: roles['harmony'].append(n)
            
        # music21 Chord analysis on harmony
        if roles['harmony']:
            midi_list = [n['midi'] for n in roles['harmony']]
            c = chord.Chord(midi_list)
            
            root = c.root().midi if c.root() else None
            third = c.third.midi if c.third else None
            fifth = c.fifth.midi if c.fifth else None
            
            diatonic_set = get_diatonic_pitches(global_key)
            
            # Priority reduction sequence inside Harmony
            kept_harmony = []
            for n in roles['harmony']:
                is_diatonic = diatonic_set is None or (n['midi'] % 12 in diatonic_set)
                
                if root and n['midi'] % 12 == root % 12:
                    n['priority'] = 3
                elif third and n['midi'] % 12 == third % 12:
                    n['priority'] = 4
                elif fifth and n['midi'] % 12 == fifth % 12:
                    n['priority'] = 5
                elif is_diatonic:
                    n['priority'] = 6 # Keep valid diatonic extensions
                else:
                    # RUTHLESS PURGE: This note is chromatic audio noise. Destroy it immediately.
                    continue 
                    
                kept_harmony.append(n)
                
            kept_harmony.sort(key=lambda x: x['priority'])
            roles['harmony'] = kept_harmony
            
        # Map Hands & Polyphony Drop based on Instrument
        time_mark = cluster_notes[0]['onset']
        
        limit_poly = 4
        two_hands = True
        
        if instrument in ["Sitar", "Flute", "Violin", "Trumpet"]:
            limit_poly = 1
            two_hands = False
            roles['harmony'] = []  # Strip harmony
            roles['bass'] = None   # Strip bass overlay
        elif instrument == "Guitar":
            limit_poly = 6
            two_hands = False
        else: # Piano defaults
            limit_poly = 4
            two_hands = True
            
        right_cnt = 0
        left_cnt = 0
        mono_cnt = 0
        
        def push_note(n, forced_hand=None):
            nonlocal right_cnt, left_cnt, mono_cnt
            
            if two_hands:
                hand = forced_hand
                if not hand:
                    hand = 'right' if n['midi'] >= 60 else 'left'
                n['hand'] = hand
                if hand == 'right' and right_cnt < limit_poly:
                    final_notes.append(n)
                    right_cnt += 1
                elif hand == 'left' and left_cnt < limit_poly:
                    final_notes.append(n)
                    left_cnt += 1
            else:
                n['hand'] = 'right'  # Map single track to 'right' channel
                if mono_cnt < limit_poly:
                    final_notes.append(n)
                    mono_cnt += 1

        # Push strict hierarchy
        if roles['melody']: push_note(roles['melody'], 'right')
        if roles['bass'] and two_hands: push_note(roles['bass'], 'left')
        elif roles['bass']: push_note(roles['bass'], 'right')
        
        for h_note in roles['harmony']:
            push_note(h_note)
            
    # Ensure impossible jumps aren't kept (Stage 3 Spatial map). 
    # Just a simple sanity pass for right hand:
    right_notes = sorted([n for n in final_notes if n['hand'] == 'right'], key=lambda x: x['onset'])
    prev = None
    cleaned_final = []
    
    for n in final_notes:
        if n['hand'] == 'right':
            if prev and abs(n['midi'] - prev['midi']) > 14 and abs(n['onset'] - prev['onset']) < 0.1:
                # Impossible instant giant jump, skip
                continue
            prev = n
        cleaned_final.append(n)
        
    print(f"    -> Outputting {len(cleaned_final)} fully arranged notes.")
    return cleaned_final


def optimize_voice_leading(arranged_notes, instrument="Piano", global_key="Ambiguous"):
    """
    Tymoczko (2006): Minimize total voice-leading distance across chord transitions.
    For each harmony note in chord N+1, check if a different octave of the same
    pitch class is closer to any note in chord N. If so, re-voice it.
    Also detects and corrects parallel 5ths/octaves.
    """
    print("[*] Optimizing voice leading across chord pairs (Tymoczko)...")
    TOL = 0.055
    lo, hi = INSTRUMENT_RANGES.get(instrument, (21, 108))

    # Group into beat clusters
    beat_clusters = []
    for n in sorted(arranged_notes, key=lambda x: x['onset']):
        placed = False
        for cluster in beat_clusters:
            if abs(n['onset'] - cluster[0]['onset']) <= TOL:
                cluster.append(n)
                placed = True
                break
        if not placed:
            beat_clusters.append([n])

    prev_cluster = None
    corrected = 0

    for cluster in beat_clusters:
        if prev_cluster is None:
            prev_cluster = cluster
            continue

        prev_midis = [n['midi'] for n in prev_cluster]

        for note in cluster:
            if note.get('role') != 'harmony':
                continue
            original_midi = note['midi']
            best_midi = original_midi
            best_dist = sum(abs(original_midi - p) for p in prev_midis)

            # Try octave up and down
            for delta in (-12, 12):
                alt = original_midi + delta
                if lo <= alt <= hi:
                    alt_dist = sum(abs(alt - p) for p in prev_midis)
                    if alt_dist < best_dist:
                        best_dist = alt_dist
                        best_midi = alt

            if best_midi != original_midi:
                note['midi'] = best_midi
                corrected += 1

        # Parallel motion detector (Tymoczko: parallel 5ths/octaves collapse voice independence)
        if len(prev_cluster) >= 2 and len(cluster) >= 2:
            for i, n1 in enumerate(cluster):
                for j, n2 in enumerate(cluster):
                    if i >= j:
                        continue
                    p1 = min(prev_cluster, key=lambda p: abs(p['midi'] - n1['midi']))
                    p2 = min(prev_cluster, key=lambda p: abs(p['midi'] - n2['midi']))

                    int_before = abs(p1['midi'] - p2['midi']) % 12
                    int_after  = abs(n1['midi'] - n2['midi']) % 12
                    move1 = n1['midi'] - p1['midi']
                    move2 = n2['midi'] - p2['midi']

                    is_parallel = (int_before == int_after and
                                   int_after in (0, 7) and
                                   np.sign(move1) == np.sign(move2) and
                                   move1 != 0)
                    if is_parallel and n2.get('role') == 'harmony':
                        fix = n2['midi'] + (12 if move2 < 0 else -12)
                        if lo <= fix <= hi:
                            print(f"  [VL] Parallel {'octave' if int_after==0 else 'fifth'} corrected: {n2['midi']} -> {fix}")
                            n2['midi'] = fix
                            corrected += 1

        prev_cluster = cluster

    print(f"[VL] Voice leading pass complete: {corrected} notes re-voiced.")
    return arranged_notes


def enforce_density_budget(notes, instrument, global_key):
    """
    Final pass: enforce per-beat note limits using ±5ms tolerance grouping.
    Tension-aware (Lerdahl): high-tension chords get +2 budget, low-tension get -1.
    Leading tone (7th scale degree) is never deleted.
    Dropout hierarchy: chromatic -> diatonic ext -> 5ths -> 3rds -> never bass/melody.
    Implied harmony purge only fires after chord identity validation.
    """
    DENSITY_LIMITS = {
        "Piano":   6,
        "Guitar":  4,
        "Sitar":   1,
        "Flute":   1,
        "Violin":  1,
        "Trumpet": 1,
    }
    base_limit   = DENSITY_LIMITS.get(instrument, 6)
    diatonic_set = get_diatonic_pitches(global_key)
    leading_tone_pc = (NOTE_MAP.get(global_key.split()[0], 0) + 11) % 12 if global_key != "Ambiguous" else None

    # Group notes into beat windows using ±5ms tolerance
    TOL = 0.005  # 5ms
    beat_groups = []  # list of lists
    for n in sorted(notes, key=lambda x: x['onset']):
        placed = False
        for group in beat_groups:
            if abs(n['onset'] - group[0]['onset']) <= TOL:
                group.append(n)
                placed = True
                break
        if not placed:
            beat_groups.append([n])

    final = []
    for group in beat_groups:
        # Lerdahl: adjust budget by tension score
        tension = compute_tension([n['midi'] for n in group], global_key)
        if tension >= 0.8:
            limit = min(base_limit + 2, 8)
        elif tension <= 0.3:
            limit = max(base_limit - 1, 2)
        else:
            limit = base_limit

        if len(group) <= limit:
            final.extend(group)
            continue

        # Separate untouchables from candidates
        locked  = [n for n in group if n.get('role') in ('bass', 'melody')]
        harmony = [n for n in group if n.get('role') not in ('bass', 'melody')]

        # Determine chord tones from music21 on the harmony block
        try:
            from music21 import chord as m21chord
            midi_list = [n['midi'] for n in harmony]
            c = m21chord.Chord(midi_list) if midi_list else None
            root_pc   = c.root().midi % 12   if (c and c.root())   else None
            third_pc  = c.third.midi % 12    if (c and c.third)    else None
            fifth_pc  = c.fifth.midi % 12    if (c and c.fifth)    else None
        except Exception:
            root_pc = third_pc = fifth_pc = None

        # Assign harmonic priority to each harmony note
        # Lower number = more important = kept last
        def harm_priority(n):
            pc = n['midi'] % 12
            is_diatonic = diatonic_set is None or pc in diatonic_set
            # Lerdahl: leading tone is structurally critical — never drop
            if leading_tone_pc is not None and pc == leading_tone_pc:
                return 0  # Absolute priority
            if root_pc is not None and pc == root_pc:
                return 1  # Root
            if third_pc is not None and pc == third_pc:
                return 2  # Third
            if fifth_pc is not None and pc == fifth_pc:
                return 3  # Fifth
            if is_diatonic:
                return 4  # Diatonic extension
            return 5      # Non-diatonic chromatic noise — drop first

        harmony.sort(key=harm_priority)

        # Implied harmony purge: if melody is a stable chord tone, 
        # try dropping the third or fifth that is already implied
        budget_remaining = limit - len(locked)
        if budget_remaining < len(harmony):
            melody_note = next((n for n in locked if n.get('role') == 'melody'), None)
            bass_note   = next((n for n in locked if n.get('role') == 'bass'), None)

            if melody_note and bass_note and diatonic_set:
                melody_pc = melody_note['midi'] % 12
                bass_pc   = bass_note['midi'] % 12

                # Check melody is an actual chord tone (not a passing tone)
                melody_is_chord_tone = (
                    (root_pc is not None and melody_pc == root_pc) or
                    (third_pc is not None and melody_pc == third_pc) or
                    (fifth_pc is not None and melody_pc == fifth_pc)
                )

                if melody_is_chord_tone:
                    # Safe to imply: drop 5th/3rd if melody+bass already outline them
                    pruned_harmony = []
                    for h in harmony:
                        pc = h['midi'] % 12
                        if fifth_pc is not None and pc == fifth_pc and melody_pc in (root_pc, third_pc):
                            print(f"  [DENSITY] Implied harmony: dropped 5th (MIDI {h['midi']}) — outlined by bass+melody.")
                            continue
                        if third_pc is not None and pc == third_pc and melody_pc == root_pc and bass_pc == root_pc:
                            print(f"  [DENSITY] Implied harmony: dropped 3rd (MIDI {h['midi']}) — bass+melody lock unison/octave.")
                            continue
                        pruned_harmony.append(h)
                    harmony = pruned_harmony

        # Keep as many harmony notes as budget allows (lowest priority dropped first = highest index)
        kept_harmony = harmony[:budget_remaining]
        dropped = harmony[budget_remaining:]
        for d in dropped:
            print(f"  [DENSITY] Budget exceeded: dropped MIDI {d['midi']} (role: harmony, priority: {harm_priority(d)})")

        final.extend(locked)
        final.extend(kept_harmony)

    print(f"[DENSITY] Budget enforcement complete. {len(final)} notes retained from {len(notes)}.")
    return final


# ─── MAIN EXECUTION ───────────────────────────────────────────────────

def arrange(json_file, audio_file, output_file, instrument="Piano"):
    print("=========================================")
    print(" EXPERT AI ARRANGER PIPELINE")
    print("=========================================\n")
    
    with open(json_file, 'r') as f:
        data = json.load(f)
        
    global_key = data.get('detected_key', 'Ambiguous')
    print(f"[*] Dynamic Profile Settings -> Instrument: {instrument}, Detected Key: {global_key}")
        
    raw_notes = []
    seen = set()
    for seg in data.get('segments', []):
        for n in seg.get('notes', []):
            n_hash = (round(n['onset'], 2), n['midi'])
            if n_hash not in seen:
                seen.add(n_hash)
                raw_notes.append(n)

    # --- GATE: Frequency Band Filter ---
    print("\n[GATE]: Instrument Physical Range Filter")
    gated_notes = frequency_band_gate(raw_notes, instrument)
        
    # --- STAGE 1 ---
    print("\n[STAGE 1]: Foundational Processing")
    cleaned = clean_notes(gated_notes, instrument=instrument)
    
    # Needs actual audio for librosa beat tracking
    quantized = adaptive_quantization(cleaned, audio_file)
    clusters = cluster_notes(quantized, window_ms=50)
    
    # --- STAGE 2 & 3 ---
    print("\n[STAGE 2/3]: Harmonics & Hand Mapping")
    assigned = assign_roles(clusters, instrument=instrument, global_key=global_key)
    arranged_notes = build_chords_and_reduce(assigned, instrument=instrument, global_key=global_key)

    # --- STAGE 3.5: Voice Leading Optimizer ---
    print("\n[STAGE 3.5]: Tymoczko Voice Leading Optimizer")
    arranged_notes = optimize_voice_leading(arranged_notes, instrument=instrument, global_key=global_key)

    # --- FINAL GATE: Density Budget (Lerdahl Tension-Aware) ---
    print("\n[STAGE 4]: Density Budget Enforcement")
    final_notes = enforce_density_budget(arranged_notes, instrument=instrument, global_key=global_key)
    
    # Format and Save
    output_data = []
    
    for n in final_notes:
        output_data.append({
            "time": n['onset'],
            "note": n['note'],
            "midi": n['midi'],
            "duration": n['duration'],
            "velocity": int(n['velocity'] * 127) if n['velocity'] <= 1.0 else int(n['velocity']),
            "hand": n.get('hand', 'right'),
            "role": n.get('role', 'harmony'),
            "phrase_start": n.get('phrase_start', False),
            "measure": 0,
            "beat": 0.0
        })
        
    # Sort chronologically
    output_data.sort(key=lambda x: x['time'])
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
        
    print(f"\n[INFO] Complete. Saved highly-structured output to {output_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="llm_input.json", help="Transcription JSON")
    parser.add_argument("--audio", default="test.mp3", help="Original Audio for beat tracking")
    parser.add_argument("--output", default="arranged.json", help="Output Arranged JSON")
    parser.add_argument("--instrument", default="Piano", help="Target instrument configuration")
    args = parser.parse_args()
    
    arrange(args.input, args.audio, args.output, instrument=args.instrument)
