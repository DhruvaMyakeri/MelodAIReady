import os
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np

# ─── CONFIG ───────────────────────────────────────────────────────────────────

SAMPLE_RATE = 22050        # Standard for audio ML
CLIP_DURATION = 3          # seconds (matches your dataset)
N_MELS = 128               # Mel filterbanks
N_FFT = 1024               # FFT window size
HOP_LENGTH = 512           # Hop between frames
TARGET_LENGTH = 128        # Fixed time frames (pad/crop to this)

# NEW — run this helper first to auto-detect your folder names
INSTRUMENTS = [
    "Accordion", "Acoustic_Guitar", "Banjo", "Bass_Guitar", "Clarinet",
    "Cymbals", "Dobro", "Drum_set", "Electro_Guitar", "Floor_Tom",
    "Harmonica", "Harmonium", "Hi_Hats", "Horn", "Keyboard", "Mandolin",
    "Organ", "Piano", "Saxophone", "Shakers", "Tambourine", "Trombone",
    "Trumpet", "Ukulele", "Violin", "cowbell", "flute", "vibraphone"
]

NUM_CLASSES = len(INSTRUMENTS)          # 28
LABEL2IDX = {name: i for i, name in enumerate(INSTRUMENTS)}
IDX2LABEL = {i: name for i, name in enumerate(INSTRUMENTS)}

# ─── DATASET CLASS ────────────────────────────────────────────────────────────

class InstrumentDataset(Dataset):
    """
    Loads audio clips, converts to Mel Spectrogram, returns (spectrogram, label).
    label is a one-hot vector of shape (NUM_CLASSES,) for multi-label support.
    """

    def __init__(self, file_paths, labels, augment=False):
        self.file_paths = file_paths
        self.labels = labels          # list of class indices
        self.augment = augment

        # Mel Spectrogram transform (runs on CPU, fast)
        self.mel_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
        )
        self.amplitude_to_db = T.AmplitudeToDB(top_db=80)

        # Augmentation transforms (only applied during training)
        self.time_masking = T.TimeMasking(time_mask_param=20)
        self.freq_masking = T.FrequencyMasking(freq_mask_param=20)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label_idx = self.labels[idx]

        # ── Load Audio ──
        # NEW
        import torchaudio
        waveform, sr = torchaudio.load(path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != SAMPLE_RATE:
            resampler = T.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)

        # ── Fix Length (pad or crop to 3 seconds) ──
        target_samples = SAMPLE_RATE * CLIP_DURATION
        if waveform.shape[1] < target_samples:
            # Pad with zeros
            pad_len = target_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        else:
            waveform = waveform[:, :target_samples]

        # ── Mel Spectrogram ──
        mel = self.mel_transform(waveform)       # (1, N_MELS, time)
        mel = self.amplitude_to_db(mel)          # Convert to dB scale

        # ── Normalize to [-1, 1] ──
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)

        # ── Fix time dimension ──
        # Pad or crop to TARGET_LENGTH frames
        if mel.shape[2] < TARGET_LENGTH:
            pad_len = TARGET_LENGTH - mel.shape[2]
            mel = torch.nn.functional.pad(mel, (0, pad_len))
        else:
            mel = mel[:, :, :TARGET_LENGTH]

        # ── Augmentation (training only) ──
        if self.augment:
            mel = self.time_masking(mel)
            mel = self.freq_masking(mel)

        # ── One-hot label ──
        label = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        label[label_idx] = 1.0

        return mel, label


# ─── DATA LOADING UTILITY ─────────────────────────────────────────────────────

def load_dataset(dataset_root: str):
    """
    Walks dataset_root, collects all audio file paths and their labels.
    Returns: (file_paths, label_indices)
    """
    file_paths = []
    label_indices = []

    for instrument_name in INSTRUMENTS:
        folder = os.path.join(dataset_root, instrument_name)
        if not os.path.isdir(folder):
            print(f"[WARNING] Folder not found: {folder}")
            continue

        files = [
            f for f in os.listdir(folder)
            if f.lower().endswith(('.mp3', '.wav', '.flac', '.ogg'))
        ]

        for fname in files:
            file_paths.append(os.path.join(folder, fname))
            label_indices.append(LABEL2IDX[instrument_name])

    print(f"[INFO] Total files loaded: {len(file_paths)}")
    print(f"[INFO] Class distribution:")
    counts = Counter(label_indices)
    for idx, count in sorted(counts.items()):
        print(f"       {IDX2LABEL[idx]:<20} {count}")

    return file_paths, label_indices


def build_dataloaders(dataset_root: str, batch_size: int = 32):
    """
    Builds train, val, and test DataLoaders with weighted sampling
    to handle class imbalance.
    """
    file_paths, labels = load_dataset(dataset_root)

    # ── Train / Val / Test Split (70 / 15 / 15) ──
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        file_paths, labels, test_size=0.30, stratify=labels, random_state=42
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, stratify=temp_labels, random_state=42
    )

    print(f"\n[INFO] Split → Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")

    # ── Weighted Sampler (fixes class imbalance in training) ──
    # Classes with fewer samples get higher sampling probability
    class_counts = Counter(train_labels)
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[lbl] for lbl in train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_paths),
        replacement=True
    )

    # ── Build Datasets ──
    train_dataset = InstrumentDataset(train_paths, train_labels, augment=True)
    val_dataset   = InstrumentDataset(val_paths,   val_labels,   augment=False)
    test_dataset  = InstrumentDataset(test_paths,  test_labels,  augment=False)

    # ── Build DataLoaders ──
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,          # Use weighted sampler instead of shuffle
        num_workers=4,
        pin_memory=True           # Faster GPU transfer
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


# ─── QUICK SANITY CHECK ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    DATASET_ROOT = "./dataset"   # <── change this to your dataset path

    train_loader, val_loader, test_loader = build_dataloaders(DATASET_ROOT, batch_size=8)

    # Grab one batch and inspect
    mel_batch, label_batch = next(iter(train_loader))
    print(f"\n[SANITY CHECK]")
    print(f"  Mel spectrogram shape : {mel_batch.shape}")   # (B, 1, 128, 128)
    print(f"  Label shape           : {label_batch.shape}") # (B, 28)

    # Visualize one spectrogram
    plt.figure(figsize=(8, 4))
    plt.imshow(mel_batch[0, 0].numpy(), aspect='auto', origin='lower', cmap='inferno')
    plt.title(f"Instrument: {IDX2LABEL[label_batch[0].argmax().item()]}")
    plt.xlabel("Time Frames")
    plt.ylabel("Mel Bins")
    plt.colorbar(label="dB")
    plt.tight_layout()
    plt.savefig("sample_mel.png")
    print("  Saved sample_mel.png")
# ```

# ---

# ### What's happening here, explained simply

# **Audio → Numbers pipeline:**
# ```
# MP3 file
#   → waveform (raw amplitude samples at 22050/sec)
#   → Mel Spectrogram (2D image: frequency bins × time frames)
#   → dB scale (log compression, matches human hearing)
#   → Normalized (zero mean, unit std)
#   → Shape: (1, 128, 128) ← this is what the CNN sees