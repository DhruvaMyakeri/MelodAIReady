import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# ─── CONFIG (must match dataset.py) ──────────────────────────────────────────
NUM_CLASSES = 28

# ─── MODEL ───────────────────────────────────────────────────────────────────

class InstrumentClassifier(nn.Module):
    """
    ResNet18 backbone pretrained on ImageNet, adapted for:
      - Single-channel input  (mel spectrogram is grayscale, not RGB)
      - Multi-label output    (sigmoid, not softmax — multiple instruments can be active)
    """

    def __init__(self, num_classes=NUM_CLASSES, dropout=0.4):
        super().__init__()

        # Load pretrained ResNet18
        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        # ── Adapt first conv layer: 3 channels → 1 channel ──
        # ResNet expects RGB (3ch), our mel spec is grayscale (1ch)
        # We average the pretrained weights across the 3 input channels
        original_weight = backbone.conv1.weight.data  # (64, 3, 7, 7)
        new_weight = original_weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)

        backbone.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64,
            kernel_size=7, stride=2, padding=3, bias=False
        )
        backbone.conv1.weight.data = new_weight

        # ── Remove the final classification head ──
        # We'll replace it with our own
        self.feature_dim = backbone.fc.in_features  # 512 for ResNet18
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # ── Custom classification head ──
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(self.feature_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes)
            # No sigmoid here — BCEWithLogitsLoss handles it numerically stable
        )

    def forward(self, x):
        # x shape: (B, 1, 128, 128)
        features = self.backbone(x)      # (B, 512)
        logits = self.classifier(features)  # (B, 28)
        return logits


# ─── MIXUP AUGMENTATION ───────────────────────────────────────────────────────

def mixup_batch(mel_batch, label_batch, alpha=0.4):
    """
    Mixup: blend two random samples in a batch.
    This simulates polyphony — two instruments playing at once.
    Returns mixed spectrograms and soft multi-label targets.
    """
    batch_size = mel_batch.size(0)
    lam = torch.distributions.Beta(alpha, alpha).sample().item()

    # Random permutation to mix with
    idx = torch.randperm(batch_size)

    mixed_mel = lam * mel_batch + (1 - lam) * mel_batch[idx]
    mixed_labels = lam * label_batch + (1 - lam) * label_batch[idx]

    return mixed_mel, mixed_labels


# ─── SANITY CHECK ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = InstrumentClassifier(num_classes=NUM_CLASSES).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Total parameters    : {total_params:,}")
    print(f"[INFO] Trainable parameters: {trainable_params:,}")

    # Dummy forward pass
    dummy_input = torch.randn(8, 1, 128, 128).to(device)
    output = model(dummy_input)
    print(f"[INFO] Input shape : {dummy_input.shape}")
    print(f"[INFO] Output shape: {output.shape}")   # Should be (8, 28)

    # Test mixup
    dummy_labels = torch.zeros(8, NUM_CLASSES)
    dummy_labels[range(8), torch.randint(0, NUM_CLASSES, (8,))] = 1.0
    mixed_mel, mixed_labels = mixup_batch(dummy_input.cpu(), dummy_labels)
    print(f"[INFO] Mixup mel shape   : {mixed_mel.shape}")
    print(f"[INFO] Mixup label shape : {mixed_labels.shape}")
    print("[OK] Model architecture looks good.")
# ```

# ---

### What's happening here

# **Why ResNet18 and not a custom CNN?**
# Pretrained weights give you feature detectors (edges, textures, patterns) for free. Training from scratch on 42k samples would take much longer to converge and likely overfit.

# **Why sigmoid and not softmax?**
# Softmax forces probabilities to sum to 1 — meaning only one instrument can "win". Sigmoid treats each class independently, so the model can say "80% Piano AND 70% Violin" simultaneously. This is what makes it multi-label.

# **What is Mixup doing?**
# ```
# Mixed spectrogram = 0.6 × Piano_clip + 0.4 × Violin_clip
# Mixed label       = [0, 0, 0, ..., 0.6(Piano), ..., 0.4(Violin), ...]