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

from src.models.eegnet_model import EEGNet
from src.datasets.seizureDatasetMulticanal import SeizureDatasetMultichannel
from src.utils.evaluador_clinico import ClinicalEvalConfig, ClinicalEvaluator


@dataclass
class TrainConfig:
    seed: int = 42
    batch_size: int = 264
    num_workers: int = 2
    max_epochs: int = 60
    patience: int = 10

    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    amp: bool = True

    label_smoothing: float = 0.0
    pos_weight_cap: float = 4.0

    sequence_length: int = 1
    augment_train: bool = False

    train_patients: tuple = (
        'chb01', 'chb02', 'chb03', 'chb04',
        'chb07', 'chb08', 'chb09', 'chb10'
    )
    val_patients: tuple = ('chb05', 'chb06')
    target_patients: tuple = (
        'chb01', 'chb02', 'chb03', 'chb04',
        'chb05', 'chb06',
        'chb07', 'chb08', 'chb09', 'chb10'
    )

    lambda_fa: float = 0.30
    min_sensitivity_gate: float = 0.20

    threshold_grid: tuple = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60)
    moving_avg_grid: tuple = (3, 5, 7, 10)

    calibr_lr: float = 1e-2
    calibr_epochs: int = 100

    save_name: str = 'eegnet_clinical_best_def.pth'


class SmoothedBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=None, smoothing=0.0):
        super().__init__()
        if pos_weight is not None:
            self.register_buffer('pos_weight', pos_weight)
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


def build_subset_dataframe(index_df, target_patients):
    df = index_df.copy()
    df['patient_id'] = df['filepath'].apply(lambda x: x.split('_')[0])
    dfs = [df[df['patient_id'] == p].copy() for p in target_patients]
    subset_df = pd.concat(dfs, axis=0).reset_index(drop=True)
    return subset_df


def make_optimizer(model, cfg):
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, cfg, ema=None):
    model.train()
    losses, probs_all, labels_all = [], [], []
    autocast_enabled = cfg.amp and device.type == 'cuda'

    for signals, labels in tqdm(loader, desc='Entrenando', leave=False):
        signals = signals.to(device, non_blocking=True)
        labels = labels.unsqueeze(1).float().to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=autocast_enabled):
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
        'loss': float(np.mean(losses)) if losses else np.nan,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'pred_pos_rate': pred_pos_rate,
        'mean_prob_pos': mean_prob_pos,
        'mean_prob_neg': mean_prob_neg,
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
        pin_memory=(device.type == 'cuda'),
        persistent_workers=num_workers > 0,
    )

    logits_all, labels_all = [], []

    for signals, labels in tqdm(loader, desc='Inferencia val', leave=False):
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


def build_val_records(subset_df, val_idx, probs, clinical_cfg):
    val_df = subset_df.loc[val_idx].copy().reset_index(drop=True)
    val_df['record_id'] = val_df['filepath'].apply(lambda x: Path(x).stem.rsplit('_win', 1)[0])
    val_df['prob'] = probs

    records = []
    helper_eval = ClinicalEvaluator(clinical_cfg)

    for record_id, group in val_df.groupby('record_id', sort=False):
        labels = group['label'].astype(int).tolist()
        ref_events = helper_eval._binary_to_events(np.asarray(labels, dtype=np.uint8))
        records.append({
            'record_id': record_id,
            'y_pred_prob': group['prob'].to_numpy(dtype=np.float32),
            'ref_events': ref_events,
            'record_duration_s': len(group) * clinical_cfg.window_step_s,
        })

    return records


def evaluate_clinical_grid(records, threshold_grid, moving_avg_grid, base_cfg, lambda_fa, min_sensitivity_gate):
    best = None

    for ma in moving_avg_grid:
        for th in threshold_grid:
            cfg = ClinicalEvalConfig(
                window_step_s=base_cfg.window_step_s,
                threshold=th,
                moving_avg_size=ma,
                min_event_duration_s=base_cfg.min_event_duration_s,
                merge_gap_s=base_cfg.merge_gap_s,
                pre_ictal_tolerance_s=base_cfg.pre_ictal_tolerance_s,
                post_ictal_tolerance_s=base_cfg.post_ictal_tolerance_s,
                max_event_duration_s=base_cfg.max_event_duration_s,
            )
            evaluator = ClinicalEvaluator(cfg)
            metrics = evaluator.evaluate_dataset(records)

            sens = metrics['event_sensitivity']
            fa_h = metrics['fa_per_hour']
            score = sens - lambda_fa * fa_h

            if sens < min_sensitivity_gate:
                score -= 1.0

            candidate = {
                'score': score,
                'threshold': th,
                'moving_avg_size': ma,
                **metrics,
            }

            if best is None or candidate['score'] > best['score']:
                best = candidate

    return best


def save_checkpoint(path, model, optimizer, epoch, best_metrics, cfg, temperature=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metrics': best_metrics,
        'config': cfg.__dict__,
    }
    if temperature is not None:
        payload['temperature'] = temperature
    torch.save(payload, path)


def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    print(f'🚀 Entrenando EEGNet en {device}')

    csv_path = os.path.join(ROOT_DIR, 'data', 'CHBMIT', 'processed', 'ventana1s', 'chbmit_index.csv')
    signals_bin = os.path.join(ROOT_DIR, 'data', 'CHBMIT', 'processed', 'ventana1s', 'chbmit_signals.bin')
    labels_bin = os.path.join(ROOT_DIR, 'data', 'CHBMIT', 'processed', 'ventana1s', 'chbmit_labels.bin')
    save_path = os.path.join(ROOT_DIR, 'models', cfg.save_name)

    index_df = pd.read_csv(csv_path)
    total_samples = len(index_df)
    subset_df = build_subset_dataframe(index_df, cfg.target_patients)

    train_idx = subset_df[subset_df['patient_id'].isin(cfg.train_patients)].index.to_numpy()
    val_idx = subset_df[subset_df['patient_id'].isin(cfg.val_patients)].index.to_numpy()

    print(f"🏥 Train patients: {cfg.train_patients} | windows={len(train_idx)}")
    print(f"🏥 Val patients: {cfg.val_patients} | windows={len(val_idx)}")

    train_labels = subset_df.loc[train_idx, 'label'].values
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
        pin_memory=(device.type == 'cuda'),
        persistent_workers=cfg.num_workers > 0,
        drop_last=False,
    )

    model = EEGNet(in_channels=18, window_size=256).to(device)

    criterion = SmoothedBCEWithLogitsLoss(
        pos_weight=pos_weight,
        smoothing=cfg.label_smoothing,
    ).to(device)

    optimizer = make_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=2,
    )
    scaler = torch.amp.GradScaler('cuda', enabled=(cfg.amp and device.type == 'cuda'))
    ema = EMA(decay=0.995)

    base_clinical_cfg = ClinicalEvalConfig(
        window_step_s=1.0,
        threshold=0.5,
        moving_avg_size=5,
        min_event_duration_s=2.0,
        merge_gap_s=30.0,
        pre_ictal_tolerance_s=0.0,
        post_ictal_tolerance_s=0.0,
        max_event_duration_s=None,
    )

    best_score = -np.inf
    patience_counter = 0

    for epoch in range(cfg.max_epochs):
        print(f'\n--- Época {epoch + 1}/{cfg.max_epochs} ---')

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
            subset_df,
            val_idx,
            val_probs,
            base_clinical_cfg,
        )

        clinical_metrics = evaluate_clinical_grid(
            records=val_records,
            threshold_grid=cfg.threshold_grid,
            moving_avg_grid=cfg.moving_avg_grid,
            base_cfg=base_clinical_cfg,
            lambda_fa=cfg.lambda_fa,
            min_sensitivity_gate=cfg.min_sensitivity_gate,
        )

        scheduler.step(clinical_metrics['score'])

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
            f"ma={clinical_metrics['moving_avg_size']} | "
            f"score={clinical_metrics['score']:.4f}"
        )

        if clinical_metrics['score'] > best_score:
            best_score = clinical_metrics['score']
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
            print('⭐ Nuevo mejor modelo guardado')
        else:
            patience_counter += 1
            print(f'⏳ Sin mejora ({patience_counter}/{cfg.patience})')

        if patience_counter >= cfg.patience:
            print('⛔ Early stopping')
            break

    print('✅ Fin entrenamiento')


if __name__ == '__main__':
    main()