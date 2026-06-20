import os
import sys
import random
from dataclasses import dataclass
from pathlib import Path
import csv

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.models.dwt_mlp_model import DWTMLP
from src.datasets.dwt_dataset import DWTSeizureDataset


@dataclass
class TrainConfig:
    seed: int = 42

    batch_size: int = 256
    num_workers: int = 0
    max_epochs: int = 100
    patience: int = 20

    lr: float = 1e-3
    weight_decay: float = 1e-3
    grad_clip_norm: float = 1.0
    amp: bool = True

    pos_weight_cap: float = 4.0

    hidden_dims: tuple = (256, 128)
    dropout: float = 0.45

    train_patients: tuple = (
        "chb01", "chb02", "chb03", "chb04",
        "chb07", "chb08", "chb09", "chb10"
    )
    val_patients: tuple = ("chb05", "chb06")

    data_dir: str = r"E:\TFM\data\CHBMIT\processed\dataset_chbmit_dwt_18ch_bin"
    save_name: str = "dwt_mlp_v2.pth"

    selection_metric: str = "pr_auc"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_index_table(index_csv_path):
    rows = []
    with open(index_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def subset_indices_by_patients(index_rows, patients):
    patients = set(patients)
    return [int(r["global_idx"]) for r in index_rows if r["patient_id"] in patients]


def compute_pos_weight(train_labels, cap_value):
    positives = int(np.sum(train_labels == 1))
    negatives = int(np.sum(train_labels == 0))

    if positives == 0:
        return torch.tensor([1.0], dtype=torch.float32)

    raw = negatives / max(positives, 1)
    clipped = min(raw, cap_value)
    print(f"⚖️ pos_weight raw={raw:.2f} | capped={clipped:.2f}")
    return torch.tensor([clipped], dtype=torch.float32)


def build_loader(dataset, batch_size, num_workers, shuffle):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )


def run_one_epoch(model, loader, criterion, device, cfg, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    losses = []
    probs_all = []
    labels_all = []

    autocast_enabled = cfg.amp and device.type == "cuda"

    context = torch.enable_grad if training else torch.no_grad

    with context():
        for features, labels, _, _ in tqdm(
            loader,
            desc="Entrenando" if training else "Validando",
            leave=False
        ):
            features = features.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=autocast_enabled
            ):
                logits = model(features)
                loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                continue

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

            probs = torch.sigmoid(logits.detach()).cpu().numpy().reshape(-1)
            labs = labels.detach().cpu().numpy().reshape(-1)

            losses.append(float(loss.detach().cpu()))
            probs_all.append(probs)
            labels_all.append(labs)

    probs = np.concatenate(probs_all) if probs_all else np.array([])
    labels = np.concatenate(labels_all) if labels_all else np.array([])

    metrics = {
        "loss": float(np.mean(losses)) if losses else np.nan,
        "pr_auc": np.nan,
        "roc_auc": np.nan,
        "mean_prob_pos": float(probs[labels == 1].mean()) if np.any(labels == 1) else 0.0,
        "mean_prob_neg": float(probs[labels == 0].mean()) if np.any(labels == 0) else 0.0,
        "pred_pos_rate": float((probs >= 0.5).mean()) if len(probs) else 0.0,
        "n_samples": int(len(labels)),
    }

    if len(probs) > 0 and len(np.unique(labels)) > 1:
        metrics["pr_auc"] = float(average_precision_score(labels, probs))
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))

    return metrics


def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)

    save_path = Path(ROOT_DIR) / "models" / cfg.save_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    index_csv_path = Path(cfg.data_dir) / "chbmit_dwt_index.csv"
    signals_path = Path(cfg.data_dir) / "chbmit_dwt_signals.bin"
    labels_path = Path(cfg.data_dir) / "chbmit_dwt_labels.bin"

    index_rows = load_index_table(index_csv_path)

    train_indices = subset_indices_by_patients(index_rows, cfg.train_patients)
    val_indices = subset_indices_by_patients(index_rows, cfg.val_patients)

    if len(train_indices) == 0:
        raise ValueError("No se encontraron índices de entrenamiento.")
    if len(val_indices) == 0:
        raise ValueError("No se encontraron índices de validación.")

    y_all = np.array([int(r["label"]) for r in index_rows], dtype=np.int32)
    train_labels = y_all[train_indices]
    pos_weight = compute_pos_weight(train_labels, cfg.pos_weight_cap).to(device)

    feature_dim = None
    for r in index_rows:
        if "feature_dim" in r and r["feature_dim"] not in (None, "", "None"):
            feature_dim = int(r["feature_dim"])
            break
    if feature_dim is None:
        feature_dim = 5184

    num_samples = len(index_rows)

    train_dataset = DWTSeizureDataset(
        signals_path=signals_path,
        labels_path=labels_path,
        index_csv_path=index_csv_path,
        indices=train_indices,
        num_samples=num_samples,
        feature_dim=feature_dim,
        normalize=True,
    )

    val_dataset = DWTSeizureDataset(
        signals_path=signals_path,
        labels_path=labels_path,
        index_csv_path=index_csv_path,
        indices=val_indices,
        num_samples=num_samples,
        feature_dim=feature_dim,
        normalize=True,
    )

    train_loader = build_loader(train_dataset, cfg.batch_size, cfg.num_workers, shuffle=True)
    val_loader = build_loader(val_dataset, cfg.batch_size, cfg.num_workers, shuffle=False)

    sample_x, _, _, _ = train_dataset[0]
    input_dim = int(sample_x.numel())

    model = DWTMLP(
        input_dim=input_dim,
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    scaler = torch.amp.GradScaler(enabled=(cfg.amp and device.type == "cuda"))

    best_metric = -1e9
    patience_counter = 0

    print("🚀 Iniciando entrenamiento DWT-MLP...\n")

    for epoch in range(1, cfg.max_epochs + 1):
        print(f"\n===== Epoch {epoch}/{cfg.max_epochs} =====")

        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            cfg=cfg,
            optimizer=optimizer,
            scaler=scaler,
        )

        val_metrics = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            cfg=cfg,
            optimizer=None,
            scaler=None,
        )

        current_metric = val_metrics[cfg.selection_metric]
        if not np.isfinite(current_metric):
            current_metric = -1e9

        scheduler.step(current_metric)

        print(
            f"Train | loss={train_metrics['loss']:.4f} | "
            f"pr_auc={train_metrics['pr_auc']:.4f} | "
            f"roc_auc={train_metrics['roc_auc']:.4f} | "
            f"pred_pos={train_metrics['pred_pos_rate']:.4f}"
        )
        print(
            f"TrainDiag | mean_prob_pos={train_metrics['mean_prob_pos']:.4f} | "
            f"mean_prob_neg={train_metrics['mean_prob_neg']:.4f}"
        )

        print(
            f"Val   | loss={val_metrics['loss']:.4f} | "
            f"pr_auc={val_metrics['pr_auc']:.4f} | "
            f"roc_auc={val_metrics['roc_auc']:.4f} | "
            f"pred_pos={val_metrics['pred_pos_rate']:.4f}"
        )
        print(
            f"ValDiag | mean_prob_pos={val_metrics['mean_prob_pos']:.4f} | "
            f"mean_prob_neg={val_metrics['mean_prob_neg']:.4f}"
        )

        if current_metric > best_metric:
            best_metric = current_metric
            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "input_dim": input_dim,
                    "feature_dim": feature_dim,
                    "config": cfg.__dict__,
                    "best_selection_metric": cfg.selection_metric,
                    "best_metric_value": best_metric,
                    "val_metrics": val_metrics,
                },
                save_path,
            )
            print(f"⭐ Nuevo mejor modelo guardado por {cfg.selection_metric}={best_metric:.4f}")
        else:
            patience_counter += 1
            print(f"⏳ Sin mejora ({patience_counter}/{cfg.patience})")

        if patience_counter >= cfg.patience:
            print("⛔ Early stopping")
            break

    print("\n✅ Fin entrenamiento")


if __name__ == "__main__":
    main()