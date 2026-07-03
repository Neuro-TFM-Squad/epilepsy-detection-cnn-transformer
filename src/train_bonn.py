import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from models.bonn import BaselineCNN
from datasets.bonn_dataset import BonnEEGDataset


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 32
    num_workers: int = 0
    max_epochs: int = 80
    patience: int = 12
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    use_amp: bool = True
    save_path: str = "models/baseline_cnn_bonn_clean.pth"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando Bonn en {device}")

    index_csv = "data/BONN/processed/bonn_index.csv"
    root_dir = "data/BONN/processed"

    dataset_aug = BonnEEGDataset(index_csv=index_csv, root_dir=root_dir, augment=True)
    dataset_clean = BonnEEGDataset(index_csv=index_csv, root_dir=root_dir, augment=False)

    labels = dataset_aug.df["label"].values
    indices = np.arange(len(dataset_aug))

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=cfg.seed,
        stratify=labels
    )

    train_dataset = Subset(dataset_aug, train_idx)
    val_dataset = Subset(dataset_clean, val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    sample_signal, _ = dataset_clean[0]
    model = BaselineCNN(in_channels=sample_signal.shape[0], num_classes=1).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.use_amp and device.type == "cuda"))

    best_val_loss = float("inf")
    patience_counter = 0
    os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)

    for epoch in range(cfg.max_epochs):
        model.train()
        train_losses = []

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.max_epochs} [train]", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.unsqueeze(1).float().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(cfg.use_amp and device.type == "cuda")
            ):
                logits = model(x)
                loss = criterion(logits, y)

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg.max_epochs} [val]", leave=False):
                x = x.to(device, non_blocking=True)
                y = y.unsqueeze(1).float().to(device, non_blocking=True)
                logits = model(x)
                loss = criterion(logits, y)
                val_losses.append(loss.item())

        mean_train = float(np.mean(train_losses)) if train_losses else np.nan
        mean_val = float(np.mean(val_losses)) if val_losses else np.nan
        print(f"Epoch {epoch+1:03d} | train_loss={mean_train:.4f} | val_loss={mean_val:.4f}")

        if mean_val < best_val_loss:
            best_val_loss = mean_val
            patience_counter = 0
            torch.save(model.state_dict(), cfg.save_path)
            print(" -> mejor modelo guardado")
        else:
            patience_counter += 1
            print(f" -> sin mejora ({patience_counter}/{cfg.patience})")

        if patience_counter >= cfg.patience:
            print("Early stopping")
            break

    print(f"Checkpoint final: {cfg.save_path}")


if __name__ == "__main__":
    main()