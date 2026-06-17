import os
import sys
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.models.hibrido import SpatioTemporalCNN
from src.datasets.seizureDatasetMulticanal import SeizureDatasetMultichannel
from src.utils.evaluador_clinico_def import ClinicalEvalConfig, ClinicalEvaluator, clinical_score
from src.utils.postprocessing import PostProcessConfig


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 264
    num_workers: int = 0
    max_epochs: int = 60
    patience: int = 10

    head_lr: float = 1e-3
    backbone_lr: float = 3e-4
    temporal_lr: float = 1e-4
    calibr_lr: float = 1e-2
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    amp: bool = True

    label_smoothing: float = 0.0
    pos_weight_cap: float = 4.0

    sequence_length: int = 1
    augment_train: bool = True
    temporal_kernel: int = 15
    temporal_filters: int = 16
    spatial_filters: int = 32
    dropout_p: float = 0.3
    spatial_dropout_p: float = 0.1
    temporal_norm: str = "instance"
    unfreeze_temporal_epoch: int = 10

    train_patients: tuple = (
        "chb01", "chb02", "chb03", "chb04",
        "chb07", "chb08", "chb09", "chb10"
    )
    val_patients: tuple = ("chb05", "chb06")
    target_patients: tuple = (
        "chb01", "chb02", "chb03", "chb04",
        "chb05", "chb06",
        "chb07", "chb08", "chb09", "chb10"
    )

    threshold_grid: tuple = (0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
    smoothing_grid: tuple = (1, 3, 5, 7)

    window_step_s: float = 1.0
    smoothing: str = "moving_average"
    use_hysteresis: bool = False
    threshold_on: float = 0.60
    threshold_off: float = 0.40
    min_event_duration_s: float = 2.0
    merge_gap_s: float = 30.0
    max_event_duration_s: float | None = None

    pre_ictal_tolerance_s: float = 0.0
    post_ictal_tolerance_s: float = 0.0
    min_overlap_s: float = 0.0

    fa_penalty: float = 0.30
    min_sensitivity_gate: float = 0.20
    latency_penalty: float = 0.0

    calibr_epochs: int = 100
    save_name: str = "spatio_temporal_cnn_clinical_best_def.pth"


class SmoothedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None, smoothing=0.0):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight = None
        self.smoothing = smoothing

    def forward(self, logits, targets):
        if self.smoothing > 0:
            targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)


class EMA:
    def __init__(self, decay=0.995):
        self.decay = decay
        self.shadow = {}

    def update(self, model):
        with torch.no_grad():
            for name, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if name not in self.shadow:
                    self.shadow[name] = p.detach().clone()
                else:
                    self.shadow[name].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def apply_to(self, model):
        backup = {}
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in self.shadow:
                    backup[name] = p.detach().clone()
                    p.copy_(self.shadow[name])
        return backup

    def restore(self, model, backup):
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in backup:
                    p.copy_(backup[name])


class TemperatureScaler(nn.Module):
    def __init__(self, init_temp=1.5):
        super().__init__()
        self.log_temp = nn.Parameter(torch.log(torch.tensor([init_temp], dtype=torch.float32)))

    def forward(self, logits):
        temperature = torch.exp(self.log_temp).clamp(min=0.5, max=10.0)
        return logits / temperature

    @property
    def temperature(self):
        return float(torch.exp(self.log_temp).detach().cpu().item())


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def safe_sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def compute_pos_weight(train_labels, cap_value):
    positives = int(np.sum(train_labels == 1))
    negatives = int(np.sum(train_labels == 0))
    if positives == 0:
        return torch.tensor([1.0], dtype=torch.float32)
    raw = negatives / max(positives, 1)
    clipped = min(raw, cap_value)
    print(f"⚖️ pos_weight raw={raw:.2f} | capped={clipped:.2f}")
    return torch.tensor([clipped], dtype=torch.float32)


def normalize_patient_id_from_filepath(x):
    return Path(str(x)).stem.split("_")[0]


def normalize_record_id_from_filepath(x):
    stem = Path(str(x)).stem
    return stem.rsplit("_win", 1)[0] if "_win" in stem else stem


def build_subset_dataframe(index_df, target_patients):
    df = index_df.copy()

    if "patient_id" not in df.columns:
        if "filepath" not in df.columns:
            raise ValueError("El índice debe contener 'patient_id' o 'filepath'.")
        df["patient_id"] = df["filepath"].apply(normalize_patient_id_from_filepath)

    df = df[df["patient_id"].isin(target_patients)].copy().reset_index(drop=True)

    if "record_id" not in df.columns:
        if "filepath" not in df.columns:
            raise ValueError("El índice debe contener 'record_id' o 'filepath'.")
        df["record_id"] = df["filepath"].apply(normalize_record_id_from_filepath)
    else:
        df["record_id"] = (
            df["record_id"]
            .astype(str)
            .str.replace(".edf", "", regex=False)
            .str.strip()
        )

    if "window_index" in df.columns:
        df["window_index"] = pd.to_numeric(df["window_index"], errors="coerce")
        df = df.dropna(subset=["window_index"]).copy()
        df["window_index"] = df["window_index"].astype(int)

    sort_cols = [c for c in ["patient_id", "record_id", "window_index"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    return df


def maybe_set_temporal_norm_eval(model):
    if hasattr(model, "bn_temp") and isinstance(model.bn_temp, nn.BatchNorm1d):
        model.bn_temp.eval()


def freeze_temporal_block(model, freeze=True):
    for name, p in model.named_parameters():
        if (
            name.startswith("temporal_conv")
            or name.startswith("temporal_shortcut")
            or name.startswith("bn_temp")
        ):
            p.requires_grad = not freeze


def inject_bonn_knowledge(chb_model, bonn_model_path, device, num_channels=18, freeze_temporal=True):
    bonn_state = torch.load(bonn_model_path, map_location=device, weights_only=False)
    state = bonn_state["model_state_dict"] if isinstance(bonn_state, dict) and "model_state_dict" in bonn_state else bonn_state

    if "block1.0.weight" not in state:
        raise KeyError("No se encontró 'block1.0.weight' en el checkpoint de Bonn.")

    bonn_conv_weights = state["block1.0.weight"]
    repeated_conv = bonn_conv_weights.repeat(num_channels, 1, 1)

    with torch.no_grad():
        chb_model.temporal_conv.weight.copy_(repeated_conv)

        if "block1.1.weight" in state and hasattr(chb_model, "bn_temp") and getattr(chb_model.bn_temp, "weight", None) is not None:
            bn_w = state["block1.1.weight"].repeat(num_channels)
            bn_b = state["block1.1.bias"].repeat(num_channels)
            if chb_model.bn_temp.weight.shape == bn_w.shape:
                chb_model.bn_temp.weight.copy_(bn_w)
                chb_model.bn_temp.bias.copy_(bn_b)

        if isinstance(chb_model.bn_temp, nn.BatchNorm1d):
            if "block1.1.running_mean" in state and "block1.1.running_var" in state:
                rm = state["block1.1.running_mean"].repeat(num_channels)
                rv = state["block1.1.running_var"].repeat(num_channels)
                if chb_model.bn_temp.running_mean.shape == rm.shape:
                    chb_model.bn_temp.running_mean.copy_(rm)
                    chb_model.bn_temp.running_var.copy_(rv)

    if freeze_temporal:
        freeze_temporal_block(chb_model, freeze=True)

    return chb_model


def make_optimizer(model, cfg, temporal_trainable=False):
    head_params, backbone_params, temporal_params = [], [], []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("fc"):
            head_params.append(p)
        elif (
            name.startswith("temporal_conv")
            or name.startswith("temporal_shortcut")
            or name.startswith("bn_temp")
        ):
            temporal_params.append(p)
        else:
            backbone_params.append(p)

    param_groups = []
    if head_params:
        param_groups.append({"params": head_params, "lr": cfg.head_lr})
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": cfg.backbone_lr})
    if temporal_trainable and temporal_params:
        param_groups.append({"params": temporal_params, "lr": cfg.temporal_lr})

    return torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, cfg, ema=None):
    model.train()
    maybe_set_temporal_norm_eval(model)
    losses, probs_all, labels_all = [], [], []
    autocast_enabled = cfg.amp and device.type == "cuda"

    for signals, labels in tqdm(loader, desc="Entrenando", leave=False):
        signals = signals.to(device, non_blocking=True)
        labels = labels.unsqueeze(1).float().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_enabled):
            logits = model(signals)
            loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()

        if ema is not None:
            ema.update(model)

        losses.append(float(loss.detach().cpu()))
        probs_all.append(torch.sigmoid(logits.detach()).float().cpu().numpy().reshape(-1))
        labels_all.append(labels.detach().cpu().numpy().reshape(-1))

    probs = np.concatenate(probs_all) if probs_all else np.array([])
    labels = np.concatenate(labels_all) if labels_all else np.array([])
    pred_pos_rate = float((probs >= 0.5).mean()) if len(probs) else 0.0

    try:
        pr_auc = average_precision_score(labels, probs) if len(labels) else 0.0
    except Exception:
        pr_auc = 0.0

    try:
        roc_auc = roc_auc_score(labels, probs) if len(labels) else 0.0
    except Exception:
        roc_auc = 0.0

    pos_mask = labels == 1
    neg_mask = labels == 0
    mean_prob_pos = float(probs[pos_mask].mean()) if np.any(pos_mask) else 0.0
    mean_prob_neg = float(probs[neg_mask].mean()) if np.any(neg_mask) else 0.0

    return {
        "loss": float(np.mean(losses)) if losses else np.nan,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "pred_pos_rate": pred_pos_rate,
        "mean_prob_pos": mean_prob_pos,
        "mean_prob_neg": mean_prob_neg,
    }


@torch.no_grad()
def collect_logits_and_labels(model, dataset, batch_size, num_workers, device, ema=None):
    backup = ema.apply_to(model) if ema is not None else None
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=num_workers > 0,
    )

    logits_all, labels_all = [], []

    for signals, labels in tqdm(loader, desc="Inferencia val", leave=False):
        signals = signals.to(device, non_blocking=True)
        logits = model(signals)
        logits_all.append(logits.cpu().numpy().reshape(-1))
        labels_all.append(labels.numpy().reshape(-1))

    if ema is not None:
        ema.restore(model, backup)

    logits_all = np.concatenate(logits_all) if logits_all else np.array([])
    labels_all = np.concatenate(labels_all) if labels_all else np.array([])

    return logits_all, labels_all


def fit_temperature_scaler(logits, labels, device, lr=1e-2, max_iter=100):
    scaler_model = TemperatureScaler(init_temp=1.5).to(device)
    optimizer = torch.optim.LBFGS(scaler_model.parameters(), lr=lr, max_iter=max_iter)
    criterion = nn.BCEWithLogitsLoss()

    logits_t = torch.tensor(logits, dtype=torch.float32, device=device).unsqueeze(1)
    labels_t = torch.tensor(labels, dtype=torch.float32, device=device).unsqueeze(1)

    def closure():
        optimizer.zero_grad()
        calibrated_logits = scaler_model(logits_t)
        loss = criterion(calibrated_logits, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler_model


def apply_temperature(logits, temp_scaler):
    with torch.no_grad():
        logits_t = torch.tensor(logits, dtype=torch.float32).unsqueeze(1)
        calibrated_logits = temp_scaler(logits_t.to(next(temp_scaler.parameters()).device))
        probs = torch.sigmoid(calibrated_logits).cpu().numpy().reshape(-1)
    return probs


def binary_to_events(binary_labels, window_step_s=1.0):
    binary_labels = np.asarray(binary_labels, dtype=np.uint8)
    events = []
    in_event = False
    start_idx = None

    for i, v in enumerate(binary_labels):
        if v == 1 and not in_event:
            in_event = True
            start_idx = i
        elif v == 0 and in_event:
            events.append((start_idx * window_step_s, i * window_step_s))
            in_event = False
            start_idx = None

    if in_event:
        events.append((start_idx * window_step_s, len(binary_labels) * window_step_s))

    return events


def build_val_records(subset_df, val_idx, probs, window_step_s=1.0):
    val_df = subset_df.loc[val_idx].copy().reset_index(drop=True)
    val_df["prob"] = probs

    records = []

    for record_id, group in val_df.groupby("record_id", sort=False):
        if "window_index" in group.columns:
            group = group.sort_values("window_index")

        labels = group["label"].astype(int).to_numpy()
        ref_events = binary_to_events(labels, window_step_s=window_step_s)

        records.append({
            "record_id": record_id,
            "y_pred_prob": group["prob"].to_numpy(dtype=np.float32),
            "ref_events": ref_events,
            "record_duration_s": len(group) * window_step_s,
        })

    return records


def evaluate_clinical_grid(records, cfg):
    best = None

    eval_cfg = ClinicalEvalConfig(
        pre_ictal_tolerance_s=cfg.pre_ictal_tolerance_s,
        post_ictal_tolerance_s=cfg.post_ictal_tolerance_s,
        min_overlap_s=cfg.min_overlap_s,
        eps=1e-8,
    )

    for smoothing_size in cfg.smoothing_grid:
        for threshold in cfg.threshold_grid:
            post_cfg = PostProcessConfig(
                window_step_s=cfg.window_step_s,
                smoothing=cfg.smoothing,
                smoothing_size=smoothing_size,
                threshold=threshold,
                use_hysteresis=cfg.use_hysteresis,
                threshold_on=cfg.threshold_on,
                threshold_off=cfg.threshold_off,
                min_event_duration_s=cfg.min_event_duration_s,
                merge_gap_s=cfg.merge_gap_s,
                max_event_duration_s=cfg.max_event_duration_s,
                clip_probs=True,
                eps=1e-8,
            )

            evaluator = ClinicalEvaluator(
                eval_config=eval_cfg,
                postprocess_config=post_cfg,
            )

            metrics = evaluator.evaluate_dataset(
                records,
                use_probabilities=True,
                return_intermediates=False,
            )

            score = clinical_score(
                metrics,
                fa_penalty=cfg.fa_penalty,
                min_sensitivity=cfg.min_sensitivity_gate,
                latency_penalty=cfg.latency_penalty,
            )

            candidate = {
                "score": score,
                "threshold": threshold,
                "smoothing_size": smoothing_size,
                **metrics,
                "eval_config": eval_cfg.to_dict(),
                "postprocess_config": post_cfg.to_dict(),
            }

            if best is None or candidate["score"] > best["score"]:
                best = candidate

    return best


def save_checkpoint(path, model, optimizer, epoch, best_metrics, cfg, temperature=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metrics": best_metrics,
        "config": cfg.__dict__,
    }
    if temperature is not None:
        payload["temperature"] = temperature
    torch.save(payload, path)


def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"🚀 Entrenando híbrido en {device}")

    csv_path = os.path.join(ROOT_DIR, "data", "CHBMIT", "processed", "ventana1s", "chbmit_index.csv")
    signals_bin = os.path.join(ROOT_DIR, "data", "CHBMIT", "processed", "ventana1s", "chbmit_signals.bin")
    labels_bin = os.path.join(ROOT_DIR, "data", "CHBMIT", "processed", "ventana1s", "chbmit_labels.bin")
    ruta_pesos_bonn = os.path.join(ROOT_DIR, "models", "baseline_cnn_bonn_clean.pth")
    save_path = os.path.join(ROOT_DIR, "models", cfg.save_name)

    index_df = pd.read_csv(csv_path)
    total_samples = len(index_df)
    subset_df = build_subset_dataframe(index_df, cfg.target_patients)

    train_idx = subset_df[subset_df["patient_id"].isin(cfg.train_patients)].index.to_numpy()
    val_idx = subset_df[subset_df["patient_id"].isin(cfg.val_patients)].index.to_numpy()

    print(f"🏥 Train patients: {cfg.train_patients} | windows={len(train_idx)}")
    print(f"🏥 Val patients: {cfg.val_patients} | windows={len(val_idx)}")

    train_labels = subset_df.loc[train_idx, "label"].values
    pos_weight = compute_pos_weight(train_labels, cfg.pos_weight_cap).to(device)

    train_dataset = SeizureDatasetMultichannel(
        signals_path=signals_bin,
        labels_path=labels_bin,
        indices=train_idx,
        augment=cfg.augment_train,
        sequence_length=cfg.sequence_length,
        num_samples=total_samples,
    )

    val_dataset = SeizureDatasetMultichannel(
        signals_path=signals_bin,
        labels_path=labels_bin,
        indices=val_idx,
        augment=False,
        sequence_length=cfg.sequence_length,
        num_samples=total_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=cfg.num_workers > 0,
        drop_last=False,
    )

    model = SpatioTemporalCNN(
        in_channels=18,
        window_size=256,
        temporal_filters=cfg.temporal_filters,
        spatial_filters=cfg.spatial_filters,
        num_classes=1,
        temporal_kernel=cfg.temporal_kernel,
        dropout_p=cfg.dropout_p,
        spatial_dropout_p=cfg.spatial_dropout_p,
        temporal_norm=cfg.temporal_norm,
    ).to(device)

    model = inject_bonn_knowledge(
        model,
        ruta_pesos_bonn,
        device,
        num_channels=18,
        freeze_temporal=True,
    )

    criterion = SmoothedBCEWithLogitsLoss(
        pos_weight=pos_weight,
        smoothing=cfg.label_smoothing,
    ).to(device)

    optimizer = make_optimizer(model, cfg, temporal_trainable=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp and device.type == "cuda"))
    ema = EMA(decay=0.995)

    best_score = -np.inf
    patience_counter = 0

    for epoch in range(cfg.max_epochs):
        print(f"\n--- Época {epoch + 1}/{cfg.max_epochs} ---")

        if epoch == cfg.unfreeze_temporal_epoch:
            freeze_temporal_block(model, freeze=False)
            optimizer = make_optimizer(model, cfg, temporal_trainable=True)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=2,
            )
            print("🔓 Descongelado fino del bloque temporal.")

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            cfg,
            ema=ema,
        )

        val_logits, val_labels = collect_logits_and_labels(
            model,
            val_dataset,
            cfg.batch_size,
            cfg.num_workers,
            device,
            ema=ema,
        )

        raw_val_probs = safe_sigmoid(val_logits)

        temp_scaler = fit_temperature_scaler(
            val_logits,
            val_labels,
            device=device,
            lr=cfg.calibr_lr,
            max_iter=cfg.calibr_epochs,
        )
        val_probs = apply_temperature(val_logits, temp_scaler)

        val_pos_mask = val_labels == 1
        val_neg_mask = val_labels == 0

        raw_mean_pos = float(raw_val_probs[val_pos_mask].mean()) if np.any(val_pos_mask) else 0.0
        raw_mean_neg = float(raw_val_probs[val_neg_mask].mean()) if np.any(val_neg_mask) else 0.0
        cal_mean_pos = float(val_probs[val_pos_mask].mean()) if np.any(val_pos_mask) else 0.0
        cal_mean_neg = float(val_probs[val_neg_mask].mean()) if np.any(val_neg_mask) else 0.0

        val_records = build_val_records(
            subset_df=subset_df,
            val_idx=val_idx,
            probs=val_probs,
            window_step_s=cfg.window_step_s,
        )

        clinical_metrics = evaluate_clinical_grid(
            records=val_records,
            cfg=cfg,
        )

        scheduler.step(clinical_metrics["score"])

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
            f"ValDiagRaw | mean_prob_pos={raw_mean_pos:.4f} | "
            f"mean_prob_neg={raw_mean_neg:.4f}"
        )
        print(
            f"ValDiagCal | mean_prob_pos={cal_mean_pos:.4f} | "
            f"mean_prob_neg={cal_mean_neg:.4f} | "
            f"T={temp_scaler.temperature:.4f}"
        )
        print(
            f"ValClin | sens={clinical_metrics['event_sensitivity']:.4f} | "
            f"fa/h={clinical_metrics['fa_per_hour']:.4f} | "
            f"f1={clinical_metrics['event_f1']:.4f} | "
            f"th={clinical_metrics['threshold']:.2f} | "
            f"smooth={clinical_metrics['smoothing_size']} | "
            f"score={clinical_metrics['score']:.4f}"
        )

        if clinical_metrics["score"] > best_score:
            best_score = clinical_metrics["score"]
            patience_counter = 0
            save_checkpoint(
                save_path,
                model,
                optimizer,
                epoch,
                clinical_metrics,
                cfg,
                temperature=temp_scaler.temperature,
            )
            print("⭐ Nuevo mejor modelo guardado")
        else:
            patience_counter += 1
            print(f"⏳ Sin mejora ({patience_counter}/{cfg.patience})")

        if patience_counter >= cfg.patience:
            print("⛔ Early stopping")
            break

    print("✅ Fin entrenamiento")


if __name__ == "__main__":
    main()