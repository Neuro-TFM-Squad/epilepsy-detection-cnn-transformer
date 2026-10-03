"""
Análisis de interpretabilidad XAI -- TFM Detección de Crisis Epilépticas
Método: Gradientes Integrados (Integrated Gradients, Sundararajan et al. 2017)
Modelos: EEGNet v2, AttentionCNN, Híbrido, DWT-MLP
Datos:   CHB-MIT chb05 (paciente de ajuste, 558 muestras positivas)
"""
# Resolve the repository for both module and direct script execution.
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import sys, warnings, numpy as np, pandas as pd, torch, pywt
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from captum.attr import IntegratedGradients

from src.models.eegnet_model import EEGNet
from src.models.attention import AttentionCNN
from src.models.hibrido import SpatioTemporalCNN
from src.models.dwt_mlp_model import DWTMLP

PROJECT  = _PROJECT_ROOT
OUT      = PROJECT / 'experiments' / 'xai'
OUT.mkdir(parents=True, exist_ok=True)
DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
N_SAMP   = 100   # muestras por clase
IG_STEPS = 30    # pasos integración (trade-off velocidad/precisión)
IG_BATCH = 8     # muestras por lote IG

CHANNELS = [
    "FP1-F7","F7-T7","T7-P7","P7-O1",
    "FP1-F3","F3-C3","C3-P3","P3-O1",
    "FP2-F4","F4-C4","C4-P4","P4-O2",
    "FP2-F8","F8-T8","T8-P8","P8-O2",
    "FZ-CZ","CZ-PZ",
]

# -- DWT level structure --------------------------------------------------------
def _dwt_slices(sig_len=256, wavelet='db4', level=5):
    coefs = pywt.wavedec(np.zeros(sig_len), wavelet, level=level)
    slices, pos = [], 0
    for c in coefs:
        slices.append((pos, pos + len(c)))
        pos += len(c)
    return slices          # [A5, D5, D4, D3, D2, D1]

DWT_SLICES    = _dwt_slices()
FEAT_PER_CH   = DWT_SLICES[-1][1]           # 288
BAND_LABELS   = ['A5 (0-4 Hz)', 'D5 (4-8 Hz)', 'D4 (8-16 Hz)',
                 'D3 (16-32 Hz)', 'D2 (32-64 Hz)', 'D1 (64-128 Hz)']
BAND_SHORT    = ['0-4', '4-8', '8-16', '16-32', '32-64', '64-128']

assert FEAT_PER_CH * 18 == 5184, "Estructura DWT inesperada"

# -- Carga de modelos -----------------------------------------------------------
def _load(ck_path, model):
    ck = torch.load(ck_path, map_location=DEVICE)
    state = ck.get('ema_state_dict', ck.get('model_state_dict'))
    model.load_state_dict(state)
    return model.eval().to(DEVICE)

def load_models():
    eeg = _load(PROJECT/'models/eegnet_pr_auc_best.pth',
                EEGNet(in_channels=18, window_size=256, num_classes=1))
    att = _load(PROJECT/'models/attention_cnn_clinical_best.pth',
                AttentionCNN(in_channels=18, num_classes=1, dropout=0.4))
    hyb = _load(PROJECT/'models/spatio_temporal_cnn_clinical_best_def.pth',
                SpatioTemporalCNN(in_channels=18, window_size=256, temporal_filters=16,
                                  spatial_filters=32, num_classes=1, temporal_kernel=15,
                                  dropout_p=0.3, spatial_dropout_p=0.1, temporal_norm='instance'))
    ck_dwt   = torch.load(PROJECT/'models/dwt_mlp_best_pr_auc.pth', map_location=DEVICE)
    input_dim = ck_dwt['input_dim']
    dwt = _load(PROJECT/'models/dwt_mlp_best_pr_auc.pth',
                DWTMLP(input_dim=input_dim, hidden_dims=(256, 128), dropout=0.25))
    return eeg, att, hyb, dwt, input_dim

# -- Carga de datos -------------------------------------------------------------
def _normalize(arr):
    """Normaliza cada muestra por su propia media y desviación."""
    m = arr.mean(axis=tuple(range(1, arr.ndim)), keepdims=True)
    s = arr.std(axis=tuple(range(1, arr.ndim)), keepdims=True)
    return (arr - m) / (s + 1e-6)

def load_raw_eeg(patient='chb05', n_pos=N_SAMP, n_neg=N_SAMP):
    """Carga ventanas EEG crudas -> (N, 18, 256)."""
    idx = pd.read_csv(PROJECT/'data/CHBMIT/processed/ventana1s/chbmit_index.csv')
    idx['_gid'] = np.arange(len(idx), dtype=np.int64)
    mask = idx['patient_id'] == patient
    pos_rows = idx[mask & (idx['label'] == 1)].head(n_pos)
    neg_rows = idx[mask & (idx['label'] == 0)].sample(n=n_neg, random_state=42)

    sigs = np.memmap(str(PROJECT/'data/CHBMIT/processed/ventana1s/chbmit_signals.bin'),
                     dtype='float32', mode='r', shape=(len(idx), 18 * 256))

    def _fetch(rows):
        x = np.array(sigs[rows['_gid'].values], dtype=np.float32)
        return _normalize(x.reshape(-1, 18, 256))

    pos_x, neg_x = _fetch(pos_rows), _fetch(neg_rows)
    print(f"Raw EEG cargado -- pos {pos_x.shape}, neg {neg_x.shape}")
    return pos_x, neg_x

def load_dwt_feats(patient='chb05', n_pos=N_SAMP, n_neg=N_SAMP, input_dim=5184):
    """Carga features DWT -> (N, 5184)."""
    idx = pd.read_csv(PROJECT/'data/CHBMIT/processed/dataset_chbmit_dwt_18ch_bin/chbmit_dwt_index.csv')
    mask = idx['patient_id'] == patient
    pos_rows = idx[mask & (idx['label'] == 1)].head(n_pos)
    neg_rows = idx[mask & (idx['label'] == 0)].sample(n=n_neg, random_state=42)

    sigs = np.memmap(str(PROJECT/'data/CHBMIT/processed/dataset_chbmit_dwt_18ch_bin/chbmit_dwt_signals.bin'),
                     dtype='float32', mode='r', shape=(len(idx), input_dim))

    def _fetch(rows):
        x = np.array(sigs[rows['global_idx'].values], dtype=np.float32)
        return _normalize(x)

    pos_x, neg_x = _fetch(pos_rows), _fetch(neg_rows)
    print(f"DWT feats cargados -- pos {pos_x.shape}, neg {neg_x.shape}")
    return pos_x, neg_x

# -- Integrated Gradients -------------------------------------------------------
def run_ig(model, x_np, n_steps=IG_STEPS, batch=IG_BATCH):
    """Ejecuta IG sobre x_np (N, ...) y devuelve atribuciones (N, ...)."""
    ig = IntegratedGradients(model)
    parts = []
    for i in range(0, len(x_np), batch):
        chunk = torch.tensor(x_np[i:i+batch], dtype=torch.float32).to(DEVICE)
        base  = torch.zeros_like(chunk)
        attr  = ig.attribute(chunk, baselines=base, n_steps=n_steps)
        parts.append(attr.detach().cpu().numpy())
        if (i // batch) % 5 == 0:
            print(f"  IG {i+len(parts[-1])}/{len(x_np)}", end='\r')
    print()
    return np.concatenate(parts, axis=0)

# -- Análisis DWT: proyección banda x canal ------------------------------------
def attr_to_bands(attr):
    """attr (N, 5184) -> importancia media (6 bandas, 18 canales)."""
    result = np.zeros((6, 18))
    for ch in range(18):
        offset = ch * FEAT_PER_CH
        for bi, (s, e) in enumerate(DWT_SLICES):
            result[bi, ch] = np.mean(np.abs(attr[:, offset+s : offset+e]))
    return result

# -- Figura 4.3: DWT análisis frecuencial --------------------------------------
def make_fig_dwt(attr_pos, attr_neg):
    bcp = attr_to_bands(attr_pos)   # (6, 18) positivos
    bcn = attr_to_bands(attr_neg)   # (6, 18) negativos

    bp_band = bcp.mean(axis=1)      # importancia media por banda (sobre canales)
    bn_band = bcn.mean(axis=1)
    bp_norm  = bp_band / (bp_band.max() + 1e-12)
    bn_norm  = bn_band / (bn_band.max() + 1e-12)

    bcp_n = bcp / (bcp.max() + 1e-12)
    bcn_n = bcn / (bcn.max() + 1e-12)
    diff  = bcp_n - bcn_n

    fig = plt.figure(figsize=(17, 5.2), facecolor='white')
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    # -- Panel (a): importancia por banda --------------------------------------
    ax = fig.add_subplot(gs[0])
    x, w = np.arange(6), 0.35
    bars_p = ax.bar(x - w/2, bp_norm, w, color='#d62728', alpha=0.88, label='Crisis')
    bars_n = ax.bar(x + w/2, bn_norm, w, color='steelblue', alpha=0.88, label='Interictal')
    ax.set_xticks(x)
    ax.set_xticklabels([f'A5\n0-4 Hz', 'D5\n4-8 Hz', 'D4\n8-16 Hz',
                         'D3\n16-32 Hz', 'D2\n32-64 Hz', 'D1\n64-128 Hz'], fontsize=8.5)
    ax.set_ylabel('Importancia media |IG| (normalizada)', fontsize=9)
    ax.set_title('(a) Importancia por banda frecuencial\n(media sobre 18 canales)', fontsize=9.5, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.18)
    ax.grid(axis='y', alpha=0.3, linewidth=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # -- Panel (b): heatmap crisis ----------------------------------------------
    ax = fig.add_subplot(gs[1])
    im = ax.imshow(bcp_n, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_yticks(range(6))
    ax.set_yticklabels(BAND_LABELS, fontsize=8)
    ax.set_xticks(range(18))
    ax.set_xticklabels(CHANNELS, rotation=90, fontsize=7)
    ax.set_title('(b) Importancia banda x canal\n(muestras de crisis, normalizada)', fontsize=9.5, fontweight='bold')
    cb = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.03)
    cb.ax.tick_params(labelsize=8)
    cb.set_label('Importancia norm.', fontsize=8)

    # -- Panel (c): diferencia discriminativa ----------------------------------
    ax = fig.add_subplot(gs[2])
    vext = max(abs(diff.min()), abs(diff.max())) + 0.03
    im2 = ax.imshow(diff, aspect='auto', cmap='RdBu_r', vmin=-vext, vmax=vext)
    ax.set_yticks(range(6))
    ax.set_yticklabels(BAND_LABELS, fontsize=8)
    ax.set_xticks(range(18))
    ax.set_xticklabels(CHANNELS, rotation=90, fontsize=7)
    ax.set_title('(c) Diferencia crisis − interictal\n(poder discriminativo por banda)', fontsize=9.5, fontweight='bold')
    cb2 = plt.colorbar(im2, ax=ax, shrink=0.75, pad=0.03)
    cb2.ax.tick_params(labelsize=8)
    cb2.set_label('Diferencia norm.', fontsize=8)

    fig.suptitle(
        'Figura 4.3 -- DWT-MLP: análisis de importancia de características por Gradientes Integrados\n'
        f'(n={N_SAMP} muestras por clase, paciente chb05)',
        fontsize=10.5, y=1.02, fontweight='bold'
    )
    path = OUT / 'fig_4_3_dwt_ig.png'
    plt.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {path}")

    # Estadísticas para el texto
    top_band = np.argmax(bp_norm)
    top_ch   = np.argmax(bcp.mean(axis=0))
    diff_band = np.argmax(diff.mean(axis=1))
    print(f"  Banda más importante (crisis): {BAND_LABELS[top_band]}")
    print(f"  Canal más importante (crisis): {CHANNELS[top_ch]}")
    print(f"  Banda más discriminativa:      {BAND_LABELS[diff_band]}")
    return str(path), top_band, top_ch, diff_band, bp_norm, bn_norm, bcp_n, bcn_n

# -- Figura 4.4: comparación modelos CNN ---------------------------------------
def make_fig_cnn(models_data):
    """
    models_data: lista de (nombre, attr_pos (N,18,256), attr_neg (N,18,256))
    """
    n = len(models_data)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3.8 * n), facecolor='white')
    if n == 1:
        axes = axes[np.newaxis, :]

    corr_vals = {}
    for mi, (name, ap, an) in enumerate(models_data):
        mp = np.mean(np.abs(ap), axis=0)   # (18, 256)
        mn = np.mean(np.abs(an), axis=0)
        vmax = max(mp.max(), mn.max())

        corr = np.corrcoef(mp.flatten(), mn.flatten())[0, 1]
        corr_vals[name] = corr

        for col, (mat, label) in enumerate([(mp, 'Crisis'), (mn, 'Interictal')]):
            ax = axes[mi, col]
            im = ax.imshow(mat, aspect='auto', cmap='hot', vmin=0, vmax=vmax)
            # Solo mostrar cada 3 etiquetas de canal para legibilidad
            tick_pos  = list(range(0, 18, 3))
            tick_labs = [CHANNELS[i] for i in tick_pos]
            ax.set_yticks(tick_pos)
            ax.set_yticklabels(tick_labs, fontsize=7.5)
            ax.set_xlabel('Tiempo (muestras, 256 Hz)', fontsize=8)
            subtitle = f'{name} -- {label}'
            if col == 0:
                subtitle += f'   [corr pos/neg = {corr:.2f}]'
            ax.set_title(subtitle, fontsize=9, fontweight='bold')
            cb = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
            cb.ax.tick_params(labelsize=7)
            cb.set_label('|IG| medio', fontsize=7.5)

    fig.suptitle(
        'Figura 4.4 -- Patrones de importancia por Gradientes Integrados: comparación entre modelos\n'
        f'(media sobre {N_SAMP} muestras por clase, paciente chb05 -- canal x tiempo)',
        fontsize=10.5, y=1.01, fontweight='bold'
    )
    plt.tight_layout()
    path = OUT / 'fig_4_4_cnn_ig.png'
    plt.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {path}")
    for name, corr in corr_vals.items():
        interp = 'BAJA discriminación' if corr > 0.85 else ('MEDIA' if corr > 0.6 else 'ALTA discriminación')
        print(f"  Correlación pos/neg {name}: {corr:.3f} ({interp})")
    return str(path), corr_vals

# -- Main -----------------------------------------------------------------------
if __name__ == '__main__':
    print(f"Dispositivo: {DEVICE}")
    print(f"DWT: {len(DWT_SLICES)} niveles, {FEAT_PER_CH} coef/canal, {FEAT_PER_CH*18} total")
    for i, (s, e) in enumerate(DWT_SLICES):
        print(f"  {BAND_LABELS[i]}: [{s}:{e}] ({e-s} coefs)")

    print("\n-- Cargando modelos -------------------------------------------------")
    eeg, att, hyb, dwt, dwt_dim = load_models()
    print("  Modelos cargados OK")

    print("\n-- Cargando datos chb05 ---------------------------------------------")
    pos_raw, neg_raw = load_raw_eeg()
    pos_dwt, neg_dwt = load_dwt_feats(input_dim=dwt_dim)

    print("\n-- Gradientes Integrados: DWT-MLP -----------------------------------")
    ig_dwt_pos = run_ig(dwt, pos_dwt)
    ig_dwt_neg = run_ig(dwt, neg_dwt)

    print("\n-- Gradientes Integrados: EEGNet v2 ---------------------------------")
    ig_eeg_pos = run_ig(eeg, pos_raw)
    ig_eeg_neg = run_ig(eeg, neg_raw)

    print("\n-- Gradientes Integrados: AttentionCNN ------------------------------")
    ig_att_pos = run_ig(att, pos_raw)
    ig_att_neg = run_ig(att, neg_raw)

    print("\n-- Gradientes Integrados: Híbrido -----------------------------------")
    ig_hyb_pos = run_ig(hyb, pos_raw)
    ig_hyb_neg = run_ig(hyb, neg_raw)

    print("\n-- Generando figuras ------------------------------------------------")
    fig43_path, top_band, top_ch, diff_band, bp_norm, bn_norm, bcp_n, bcn_n = \
        make_fig_dwt(ig_dwt_pos, ig_dwt_neg)

    fig44_path, corr_vals = make_fig_cnn([
        ('EEGNet v2',    ig_eeg_pos, ig_eeg_neg),
        ('AttentionCNN', ig_att_pos, ig_att_neg),
        ('Híbrido',      ig_hyb_pos, ig_hyb_neg),
    ])

    # Guardar atribuciones para referencia
    np.save(str(OUT / 'attr_dwt_pos.npy'), ig_dwt_pos)
    np.save(str(OUT / 'attr_dwt_neg.npy'), ig_dwt_neg)
    np.save(str(OUT / 'attr_eeg_pos.npy'), ig_eeg_pos)
    np.save(str(OUT / 'attr_eeg_neg.npy'), ig_eeg_neg)
    np.save(str(OUT / 'attr_att_pos.npy'), ig_att_pos)
    np.save(str(OUT / 'attr_att_neg.npy'), ig_att_neg)
    np.save(str(OUT / 'attr_hyb_pos.npy'), ig_hyb_pos)
    np.save(str(OUT / 'attr_hyb_neg.npy'), ig_hyb_neg)

    print("\n-- Resumen para la memoria ------------------------------------------")
    print(f"Figura 4.3: {fig43_path}")
    print(f"Figura 4.4: {fig44_path}")
    print(f"\nDWT banda más importante: {BAND_LABELS[top_band]}")
    print(f"DWT banda más discriminativa: {BAND_LABELS[diff_band]}")
    print(f"Canal más relevante en crisis: {CHANNELS[top_ch]}")
    print("\nCorrelación positivo/negativo por modelo CNN:")
    for name, c in corr_vals.items():
        print(f"  {name}: {c:.3f}")
    print("\nAnálisis XAI completado.")
