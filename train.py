import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm

from dataset import build_dataloaders, NUM_CLASSES, IDX2LABEL
from model import InstrumentClassifier, mixup_batch

# ─── CONFIG ───────────────────────────────────────────────────────────────────

DATASET_ROOT  = "./dataset"
BATCH_SIZE    = 32
NUM_EPOCHS    = 30
LEARNING_RATE = 3e-4
MIXUP_PROB    = 0.5
THRESHOLD     = 0.3   # 🔥 changed from 0.5 → better for multi-label
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ─── UTILITIES ────────────────────────────────────────────────────────────────

def compute_f1(logits, targets, threshold=THRESHOLD):
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= threshold).astype(int)
    targets = targets.cpu().numpy().astype(int)
    return f1_score(targets, preds, average="macro", zero_division=0)


def compute_class_weights(loader, num_classes=NUM_CLASSES):
    pos_counts = torch.zeros(num_classes)
    total = 0

    for _, labels in loader:
        pos_counts += labels.sum(dim=0)
        total += labels.shape[0]

    weights = total / (num_classes * pos_counts.clamp(min=1))
    return weights


# ─── TRAIN ────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, scheduler, device, epoch):
    model.train()
    total_loss = 0.0
    all_logits, all_targets = [], []

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [Train]", leave=False)

    for mel, labels in pbar:
        mel, labels = mel.to(device), labels.to(device)

        # Mixup
        if torch.rand(1).item() < MIXUP_PROB:
            mel, labels = mixup_batch(mel, labels)
            mel, labels = mel.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(mel)
        loss = criterion(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        all_logits.append(logits.detach().cpu())
        all_targets.append(labels.detach().cpu())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader)
    f1 = compute_f1(torch.cat(all_logits), torch.cat(all_targets))

    return avg_loss, f1


# ─── VALIDATION ───────────────────────────────────────────────────────────────

def val_epoch(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = 0.0
    all_logits, all_targets = [], []

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [Val]", leave=False)

        for mel, labels in pbar:
            mel, labels = mel.to(device), labels.to(device)

            logits = model(mel)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            all_logits.append(logits.cpu())
            all_targets.append(labels.cpu())

    avg_loss = total_loss / len(loader)
    f1 = compute_f1(torch.cat(all_logits), torch.cat(all_targets))

    # Per-class F1
    probs = torch.sigmoid(torch.cat(all_logits)).numpy()
    preds = (probs >= THRESHOLD).astype(int)
    targets = torch.cat(all_targets).numpy().astype(int)

    per_class_f1 = f1_score(targets, preds, average=None, zero_division=0)

    return avg_loss, f1, per_class_f1


# ─── MAIN TRAIN LOOP ──────────────────────────────────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training on: {device}")

    # Data
    train_loader, val_loader, test_loader = build_dataloaders(
        DATASET_ROOT, BATCH_SIZE
    )

    # Model
    model = InstrumentClassifier(num_classes=NUM_CLASSES).to(device)
    print(f"[INFO] Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Loss
    print("[INFO] Computing class weights...")
    class_weights = compute_class_weights(train_loader).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Scheduler
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=NUM_EPOCHS,
        pct_start=0.1
    )

    # Logging
    writer = SummaryWriter("runs/instrument_classifier")

    best_val_f1 = 0.0
    patience = 7
    patience_counter = 0

    print("\n" + "="*60)
    print(f"Starting training for {NUM_EPOCHS} epochs")
    print("="*60 + "\n")

    for epoch in range(1, NUM_EPOCHS + 1):

        train_loss, train_f1 = train_epoch(
            model, train_loader, optimizer, criterion, scheduler, device, epoch
        )

        val_loss, val_f1, per_class_f1 = val_epoch(
            model, val_loader, criterion, device, epoch
        )

        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("F1", {"train": train_f1, "val": val_f1}, epoch)

        print(f"Epoch {epoch:02d}/{NUM_EPOCHS} │ "
              f"Train Loss: {train_loss:.4f} F1: {train_f1:.4f} │ "
              f"Val Loss: {val_loss:.4f} F1: {val_f1:.4f}")

        print(f"  Best Val F1 so far: {best_val_f1:.4f}")

        # Per-class insight
        if epoch % 5 == 0:
            print("\nPer-class F1:")
            for i, f1 in enumerate(per_class_f1):
                bar = "█" * int(f1 * 20)
                print(f"{IDX2LABEL[i]:<20} {f1:.3f} {bar}")
            print()

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
            }, os.path.join(CHECKPOINT_DIR, "best_model.pt"))

            print(f"  ✓ Saved best model (F1: {val_f1:.4f})")

        else:
            patience_counter += 1

            if patience_counter >= patience:
                print("\n[INFO] Early stopping triggered")
                break

    print(f"\n[DONE] Best Val F1: {best_val_f1:.4f}")
    writer.close()


# ─── ENTRY ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    train()