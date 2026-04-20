import librosa
import numpy as np


NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

def detect_pitch(y, sr):
    """
    Returns:
        dominant frequency + note
    """

    # Pitch tracking
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    pitch_values = []

    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]

        if pitch > 0:
            pitch_values.append(pitch)

    if len(pitch_values) == 0:
        return None, None

    dominant_freq = np.median(pitch_values)

    # Convert frequency → note
    note_index = int(round(12 * np.log2(dominant_freq / 440.0) + 69))
    note_name = NOTE_NAMES[note_index % 12]

    return float(dominant_freq), note_name
# ─── MAIN FEATURE EXTRACTION ─────────────────────────────

def extract_features(y, sr):
    """
    Input:
        y  → audio waveform (numpy array)
        sr → sample rate

    Output:
        dict of audio features
    """

    features = {}

    # ─── MFCC (TIMBRE) ─────────────────────────────
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    features["mfcc_mean"] = np.mean(mfcc, axis=1).tolist()
    features["mfcc_var"]  = np.var(mfcc, axis=1).tolist()

    # ─── TEMPO (BEATS) ─────────────────────────────
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features["tempo"] = float(tempo)

    # ─── ENERGY (LOUDNESS) ─────────────────────────
    rms = librosa.feature.rms(y=y)
    features["energy"] = float(np.mean(rms))

    # ─── SPECTRAL CENTROID (BRIGHTNESS) ────────────
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features["spectral_centroid"] = float(np.mean(centroid))

    # ─── ZERO CROSSING RATE (TEXTURE) ──────────────
    zcr = librosa.feature.zero_crossing_rate(y)
    features["zero_crossing_rate"] = float(np.mean(zcr))

    # ─── CHROMA (PITCH / KEY HINT) ─────────────────
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    # 12 bins → C, C#, D, ..., B
    features["chroma_mean"] = np.mean(chroma, axis=1).tolist()

    # ─── OPTIONAL: DOMINANT FREQUENCY ──────────────
    spectrum = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    dominant_freq = freqs[np.argmax(np.mean(spectrum, axis=1))]
    features["dominant_frequency"] = float(dominant_freq)
    freq, note = detect_pitch(y, sr)

    features["dominant_frequency"] = freq
    features["detected_note"] = note
    return features