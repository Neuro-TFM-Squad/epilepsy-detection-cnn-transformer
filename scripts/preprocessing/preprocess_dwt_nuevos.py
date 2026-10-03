"""
Genera features DWT para pacientes nuevos (chb11-chb15 o cualquier lista),
escribiendo en un directorio separado para no sobreescribir el dataset original.
Los global_idx son relativos al nuevo dataset (empiezan en 0).
"""
# Resolve the repository for both module and direct script execution.
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))


import os
import csv
import numpy as np
import pywt
import mne
from tqdm import tqdm

# ==========================================
# Configuración
# ==========================================
RUTA_DATASET = str(_PROJECT_ROOT / "data/CHBMIT/raw")
PACIENTES = ["chb11", "chb12", "chb13", "chb15"]

WAVELET = "db4"
NIVEL_DESCOMP = 5
FS = 256
TAM_VENTANA = FS * 1
CARPETA_SALIDA = str(_PROJECT_ROOT / "data/CHBMIT/processed/dataset_chbmit_dwt_nuevos")

SOTA_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ"
]

os.makedirs(CARPETA_SALIDA, exist_ok=True)

BIN_SIGNALS = os.path.join(CARPETA_SALIDA, "chbmit_dwt_signals.bin")
BIN_LABELS = os.path.join(CARPETA_SALIDA, "chbmit_dwt_labels.bin")
INDEX_CSV = os.path.join(CARPETA_SALIDA, "chbmit_dwt_index.csv")


# ==========================================
# Funciones auxiliares
# ==========================================
def parsear_summary(ruta_summary):
    ataques = {}
    archivo_actual = None
    inicio = None
    with open(ruta_summary, "r") as f:
        for linea in f:
            linea = linea.strip()
            if linea.startswith("File Name:"):
                archivo_actual = linea.split(":", 1)[1].strip()
                if archivo_actual not in ataques:
                    ataques[archivo_actual] = []
            elif linea.startswith("Seizure") and "Start Time" in linea:
                inicio = int(linea.split(":", 1)[1].strip().replace(" seconds", ""))
            elif linea.startswith("Seizure") and "End Time" in linea:
                fin = int(linea.split(":", 1)[1].strip().replace(" seconds", ""))
                if archivo_actual is not None and inicio is not None:
                    ataques[archivo_actual].append((inicio, fin))
                inicio = None
    return ataques


def armonizar_canales(raw, target_channels):
    rename_dict = {}
    channels_to_drop = []
    existing_clean = set(raw.ch_names).intersection(set(target_channels))
    for ch in raw.ch_names:
        if ch.endswith("-0") or ch.endswith("-1"):
            clean = ch[:-2]
            if clean in target_channels:
                if clean in existing_clean or clean in rename_dict.values():
                    channels_to_drop.append(ch)
                else:
                    rename_dict[ch] = clean
    if channels_to_drop:
        raw.drop_channels(channels_to_drop)
    if rename_dict:
        raw.rename_channels(rename_dict)
    raw.pick_channels(target_channels)
    raw.reorder_channels(target_channels)
    return raw


def extraer_coeficientes_dwt(ventana):
    coefs = []
    for canal in ventana:
        coefs_canal = np.concatenate(pywt.wavedec(canal, WAVELET, level=NIVEL_DESCOMP))
        coefs.append(coefs_canal)
    return np.concatenate(coefs).astype(np.float32)


def procesar_edf(ruta_edf, nombre_edf, ataques_por_archivo):
    raw = mne.io.read_raw_edf(ruta_edf, preload=True, verbose="ERROR")
    try:
        raw = armonizar_canales(raw, SOTA_CHANNELS)
    except Exception:
        del raw
        return None, None
    if int(raw.info["sfreq"]) != FS:
        raw.resample(FS, npad="auto")
    senal = raw.get_data().astype(np.float32)
    n_muestras = senal.shape[1]
    intervalos = []
    if nombre_edf in ataques_por_archivo:
        for s, e in ataques_por_archivo[nombre_edf]:
            intervalos.append((s * FS, e * FS))
    n_ventanas = n_muestras // TAM_VENTANA
    X_lista, y_lista = [], []
    for v in range(n_ventanas):
        ini = v * TAM_VENTANA
        fin = ini + TAM_VENTANA
        ventana = senal[:, ini:fin]
        if ventana.shape != (18, TAM_VENTANA):
            continue
        label = 0
        for ia, fa in intervalos:
            if ini < fa and fin > ia:
                label = 1
                break
        X_lista.append(extraer_coeficientes_dwt(ventana))
        y_lista.append(label)
    del raw, senal
    if not X_lista:
        return None, None
    return np.asarray(X_lista, dtype=np.float32), np.asarray(y_lista, dtype=np.int32)


# ==========================================
# Procesamiento principal
# ==========================================
print(f"Procesando pacientes: {PACIENTES}")
print(f"Salida: {CARPETA_SALIDA}\n")

index_rows = []
feature_dim = None
global_idx = 0
resumen = []

with open(BIN_SIGNALS, "wb") as f_sig, open(BIN_LABELS, "wb") as f_lab:
    for paciente in tqdm(PACIENTES, desc="Pacientes"):
        ruta_pac = os.path.join(RUTA_DATASET, paciente)
        if not os.path.exists(ruta_pac):
            print(f"No encontrado: {paciente}")
            continue
        ruta_summary = os.path.join(ruta_pac, f"{paciente}-summary.txt")
        if not os.path.exists(ruta_summary):
            print(f"Sin summary: {paciente}")
            continue
        ataques = parsear_summary(ruta_summary)
        edfs = sorted([f for f in os.listdir(ruta_pac) if f.lower().endswith(".edf")])
        total_v, total_a, ok, omit = 0, 0, 0, 0
        for edf in tqdm(edfs, desc=paciente, leave=False):
            ruta_edf = os.path.join(ruta_pac, edf)
            nombre_base = os.path.splitext(edf)[0]
            try:
                X, y = procesar_edf(ruta_edf, edf, ataques)
                if X is None:
                    omit += 1
                    continue
                if feature_dim is None:
                    feature_dim = X.shape[1]
                if X.shape[1] != feature_dim:
                    omit += 1
                    continue
                X.tofile(f_sig)
                y.astype(np.float32).tofile(f_lab)
                for i in range(len(y)):
                    index_rows.append([global_idx, paciente, nombre_base, i, int(y[i])])
                    global_idx += 1
                total_v += len(y)
                total_a += int(y.sum())
                ok += 1
                del X, y
            except Exception as e:
                print(f"Error {edf}: {e}")
                omit += 1
        resumen.append((paciente, total_v, total_a, ok, omit))
        print(f"{paciente}: {total_v} ventanas | {total_a} crisis | {ok} EDFs OK | {omit} omitidos")

with open(INDEX_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["global_idx", "patient_id", "record_id", "window_idx", "label"])
    writer.writerows(index_rows)

print("\n=== RESUMEN ===")
for pac, v, a, ok, omit in resumen:
    print(f"{pac}: {v} ventanas | {a} crisis | {ok} OK | {omit} omitidos")
print(f"\nTotal ventanas: {sum(r[1] for r in resumen)}")
print(f"Feature dim: {feature_dim}")
print(f"Index CSV: {INDEX_CSV}")
