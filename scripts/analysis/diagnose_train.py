"""
Diagnóstico rápido: ¿los modelos detectan crisis en sus propios pacientes de entrenamiento?
Ejecutar desde la raíz con: python scripts/analysis/diagnose_train.py
"""
# Resolve the repository for both module and direct script execution.
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import sys, json, os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from pathlib import Path

from src.datasets.seizureDatasetMulticanal import SeizureDatasetMultichannel
from src.datasets.dwt_dataset import DWTSeizureDataset
from src.models.eegnet_model import EEGNet
from src.models.hibrido import SpatioTemporalCNN
from src.models.attention import AttentionCNN
from src.models.dwt_mlp_model import DWTMLP
from src.utils.postprocessing import PostProcessConfig, PredictionPostProcessor
from src.utils.evaluador_clinico_def import ClinicalEvaluator, ClinicalEvalConfig, clinical_score

ROOT = _PROJECT_ROOT
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAIN_PATIENTS = ['chb01', 'chb03', 'chb15']  # rich seizure records, never seen in validation
VAL_PATIENTS   = ['chb05', 'chb06']

# Best postprocessing configs from final evaluation (tune=chb05)
PP_CONFIGS = {
    'eegnet':               PostProcessConfig(smoothing='moving_average', smoothing_size=1, threshold=0.20, min_event_duration_s=5.0, merge_gap_s=0.0),
    'attention_cnn':        PostProcessConfig(smoothing='moving_average', smoothing_size=5, threshold=0.50, min_event_duration_s=5.0, merge_gap_s=5.0),
    'spatio_temporal_cnn':  PostProcessConfig(smoothing='moving_average', smoothing_size=1, threshold=0.30, min_event_duration_s=1.0, merge_gap_s=5.0),
    'dwt':                  PostProcessConfig(smoothing='moving_average', smoothing_size=3, threshold=0.30, min_event_duration_s=5.0, merge_gap_s=0.0),
}

CHECKPOINTS = {
    'eegnet':               ROOT / 'models/eegnet_clinical_best_def.pth',
    'attention_cnn':        ROOT / 'models/attention_cnn_clinical_best.pth',
    'spatio_temporal_cnn':  ROOT / 'models/spatio_temporal_cnn_clinical_best_def.pth',
    'dwt':                  ROOT / 'models/dwt_mlp_best_pr_auc.pth',
}

def build_model(name, ck):
    if name == 'eegnet':
        m = EEGNet(in_channels=18, window_size=256, num_classes=1)
    elif name == 'attention_cnn':
        m = AttentionCNN(in_channels=18, num_classes=1, dropout=0.4)
    elif name == 'spatio_temporal_cnn':
        m = SpatioTemporalCNN(in_channels=18, window_size=256, temporal_filters=16,
                               spatial_filters=32, num_classes=1, temporal_kernel=15,
                               dropout_p=0.3, spatial_dropout_p=0.1, temporal_norm='instance')
    elif name == 'dwt':
        cfg = ck.get('config', {})
        hidden_dims = tuple(cfg.get('hidden_dims', (256, 128)))
        m = DWTMLP(input_dim=ck['input_dim'], hidden_dims=hidden_dims, dropout=cfg.get('dropout', 0.25))
    state = ck if not isinstance(ck, dict) else ck.get('model_state_dict', ck)
    m.load_state_dict(state)
    m.to(DEVICE)
    m.eval()
    return m

def build_ref_events(group):
    events, in_ev, ev_s, ev_e = [], False, None, None
    for _, row in group.iterrows():
        if row['label'] == 1 and not in_ev:
            in_ev, ev_s, ev_e = True, float(row['start_sec']), float(row['end_sec'])
        elif row['label'] == 1 and in_ev:
            ev_e = max(ev_e, float(row['end_sec']))
        elif row['label'] == 0 and in_ev:
            events.append((ev_s, ev_e)); in_ev = False
    if in_ev: events.append((ev_s, ev_e))
    return events

@torch.no_grad()
def predict_raw(model, name, ck, patients):
    """Run inference and return {record_id: probs_array}"""
    results = {}

    if name == 'dwt':
        data_root = ROOT / 'data/CHBMIT/processed/dataset_chbmit_dwt_18ch_bin'
        csv_path  = data_root / 'chbmit_dwt_index.csv'
        idx_df    = pd.read_csv(csv_path)
        sub_df    = idx_df[idx_df['patient_id'].isin(patients)].reset_index(drop=True)
        global_ix = sub_df['global_idx'].astype(int).tolist()
        feat_dim  = ck.get('feature_dim', ck.get('config', {}).get('feature_dim', 5184))
        ds = DWTSeizureDataset(
            signals_path=str(data_root / 'chbmit_dwt_signals.bin'),
            labels_path =str(data_root / 'chbmit_dwt_labels.bin'),
            index_csv_path=str(csv_path),
            indices=global_ix, num_samples=len(idx_df),
            feature_dim=feat_dim, normalize=True,
        )
        loader = DataLoader(Subset(ds, list(range(len(global_ix)))),
                            batch_size=512, shuffle=False, num_workers=0)
        all_probs = []
        for batch in loader:
            x = batch[0].to(DEVICE)
            p = torch.sigmoid(model(x)).cpu().numpy().reshape(-1)
            all_probs.append(p)
        all_probs = np.concatenate(all_probs)
        sub_df = sub_df.copy()
        sub_df['prob'] = all_probs
        for rid, grp in sub_df.groupby('record_id'):
            results[rid] = grp.sort_values('global_idx')['prob'].values
        return results, sub_df

    # Raw-signal models
    data_root = ROOT / 'data/CHBMIT/processed/ventana1s'
    csv_path  = data_root / 'chbmit_index.csv'  # FULL index
    idx_df    = pd.read_csv(csv_path)
    sub_df    = idx_df[idx_df['patient_id'].isin(patients)].sort_values(
                    ['patient_id','record_id','window_index']).reset_index(drop=True)
    n_full    = len(idx_df)
    local_ix  = sub_df.index.tolist()
    ds = SeizureDatasetMultichannel(
        signals_path=str(data_root / 'chbmit_signals.bin'),
        labels_path =str(data_root / 'chbmit_labels.bin'),
        augment=False, sequence_length=1,
        indices=local_ix, num_samples=n_full,
    )
    loader = DataLoader(Subset(ds, list(range(len(local_ix)))),
                        batch_size=512, shuffle=False, num_workers=0)
    all_probs = []
    for batch in loader:
        x = batch[0].to(DEVICE)
        p = torch.sigmoid(model(x)).cpu().numpy().reshape(-1)
        all_probs.append(p)
    all_probs = np.concatenate(all_probs)
    sub_df = sub_df.copy()
    sub_df['prob'] = all_probs
    for rid, grp in sub_df.groupby('record_id'):
        results[rid] = grp.sort_values('window_index')['prob'].values
    return results, sub_df

def eval_clinical(probs_dict, ref_df_patients, pp_cfg):
    eval_cfg = ClinicalEvalConfig()
    evaluator = ClinicalEvaluator(eval_config=eval_cfg, postprocess_config=pp_cfg)

    total_tp = total_fp = total_fn = 0
    total_dur = 0.0
    record_results = []

    for patient in ref_df_patients:
        pat_df = ref_df_patients[patient]
        for rid, grp in pat_df.groupby('record_id'):
            grp = grp.sort_values('window_index')
            ref_events = build_ref_events(grp)
            dur = float(grp['end_sec'].max() - grp['start_sec'].min())
            if rid not in probs_dict:
                continue
            probs = probs_dict[rid]
            res = evaluator.evaluate_record_from_events(
                pred_events=evaluator.events_from_probabilities(probs),
                ref_events=ref_events,
                record_duration_s=dur,
                record_id=rid,
            )
            total_tp += res['tp_event']
            total_fp += res['fp_event']
            total_fn += res['fn_event']
            total_dur += dur
            if ref_events:
                record_results.append({
                    'rid': rid, 'n_ref': len(ref_events),
                    'tp': res['tp_event'], 'fp': res['fp_event'], 'fn': res['fn_event'],
                    'sens': res['event_sensitivity'], 'fa_h': res['fa_per_hour'],
                })

    eps = 1e-8
    sens = total_tp / (total_tp + total_fn + eps)
    prec = total_tp / (total_tp + total_fp + eps)
    f1   = 2 * sens * prec / (sens + prec + eps)
    fa_h = total_fp / (total_dur / 3600.0 + eps)

    return {
        'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
        'sens': sens, 'prec': prec, 'f1': f1, 'fa_h': fa_h,
        'per_record': record_results,
    }

# ---- Main ----
full_idx = pd.read_csv(ROOT / 'data/CHBMIT/processed/ventana1s/chbmit_index.csv')

# Split into train vs val
train_df = full_idx[full_idx['patient_id'].isin(TRAIN_PATIENTS)]
val_df   = full_idx[full_idx['patient_id'].isin(VAL_PATIENTS)]

train_by_patient = {p: train_df[train_df['patient_id'] == p] for p in TRAIN_PATIENTS}
val_by_patient   = {p: val_df[val_df['patient_id'] == p]     for p in VAL_PATIENTS}

print(f"Device: {DEVICE}\n")
print(f"{'Model':<22} {'TRAIN Sens':>10} {'TRAIN FA/h':>10} {'TRAIN TP/tot':>12} {'VAL Sens':>9} {'VAL FA/h':>9} {'VAL TP/tot':>11}")
print('-' * 90)

all_results = {}
for name in ['eegnet', 'attention_cnn', 'spatio_temporal_cnn', 'dwt']:
    ck = torch.load(CHECKPOINTS[name], map_location=DEVICE, weights_only=False)
    model = build_model(name, ck)
    pp = PP_CONFIGS[name]

    # Run on train patients
    train_probs, _ = predict_raw(model, name, ck, TRAIN_PATIENTS)
    # Run on val patients
    val_probs, _   = predict_raw(model, name, ck, VAL_PATIENTS)

    tr = eval_clinical(train_probs, train_by_patient, pp)
    va = eval_clinical(val_probs,   val_by_patient,   pp)

    tot_tr_ev = tr['tp'] + tr['fn']
    tot_va_ev = va['tp'] + va['fn']

    print(f"{name:<22} {tr['sens']:>10.3f} {tr['fa_h']:>10.2f} {tr['tp']:>5}/{tot_tr_ev:<6} "
          f"{va['sens']:>9.3f} {va['fa_h']:>9.2f} {va['tp']:>5}/{tot_va_ev:<5}")

    all_results[name] = {'train': tr, 'val': va}

print('\n\n=== DETALLE POR REGISTRO (solo con crisis) ===')
for name, res in all_results.items():
    print(f'\n--- {name} TRAIN ({TRAIN_PATIENTS}) ---')
    for r in res['train']['per_record']:
        print(f"  {r['rid']}: {r['tp']}/{r['n_ref']} detected, FA/h={r['fa_h']:.1f}")
    print(f'--- {name} VAL ---')
    for r in res['val']['per_record']:
        print(f"  {r['rid']}: {r['tp']}/{r['n_ref']} detected, FA/h={r['fa_h']:.1f}")

print('\nDiagnóstico completo.')
