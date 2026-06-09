import os
import csv
import numpy as np
import pywt
import mne
from tqdm import tqdm

# ==========================================
# 1. Configuración
# ==========================================
ruta_dataset = r"E:\TFM\data\CHBMIT\raw"
pacientes = [f"chb{str(i).zfill(2)}" for i in range(1, 11)]

wavelet = "db4"
nivel_descomp = 5
fs = 256
tam_ventana_mues = fs * 1
carpeta_salida = r"E:\TFM\data\CHBMIT\processed\dataset_chbmit_dwt_18ch_bin"

SOTA_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ"
]

os.makedirs(carpeta_salida, exist_ok=True)

bin_signals_path = os.path.join(carpeta_salida, "chbmit_dwt_signals.bin")
bin_labels_path = os.path.join(carpeta_salida, "chbmit_dwt_labels.bin")
index_csv_path = os.path.join(carpeta_salida, "chbmit_dwt_index.csv")


# ==========================================
# 2. Funciones auxiliares
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

    existing_clean_channels = set(raw.ch_names).intersection(set(target_channels))

    for ch in raw.ch_names:
        if ch.endswith("-0") or ch.endswith("-1"):
            clean_name = ch[:-2]
            if clean_name in target_channels:
                if clean_name in existing_clean_channels or clean_name in rename_dict.values():
                    channels_to_drop.append(ch)
                else:
                    rename_dict[ch] = clean_name

    if channels_to_drop:
        raw.drop_channels(channels_to_drop)

    if rename_dict:
        raw.rename_channels(rename_dict)

    raw.pick_channels(target_channels)
    raw.reorder_channels(target_channels)
    return raw


def extraer_coeficientes_dwt(ventana):
    coefs_ventana = []
    for canal in ventana:
        coeficientes = pywt.wavedec(canal, wavelet, level=nivel_descomp)
        coefs_canal = np.concatenate(coeficientes)
        coefs_ventana.append(coefs_canal)
    return np.concatenate(coefs_ventana).astype(np.float32)


def procesar_edf(ruta_edf, nombre_edf, ataques_por_archivo):
    raw = mne.io.read_raw_edf(ruta_edf, preload=True, verbose="ERROR")

    try:
        raw = armonizar_canales(raw, SOTA_CHANNELS)
    except Exception:
        del raw
        return None, None, None

    if int(raw.info["sfreq"]) != fs:
        raw.resample(fs, npad="auto")

    senal = raw.get_data().astype(np.float32)
    _, n_muestras = senal.shape

    intervalos_ataque = []
    if nombre_edf in ataques_por_archivo:
        for inicio_s, fin_s in ataques_por_archivo[nombre_edf]:
            intervalos_ataque.append((inicio_s * fs, fin_s * fs))

    n_ventanas = n_muestras // tam_ventana_mues
    X_lista, y_lista = [], []

    for v in range(n_ventanas):
        inicio_v = v * tam_ventana_mues
        fin_v = inicio_v + tam_ventana_mues
        ventana = senal[:, inicio_v:fin_v]

        if ventana.shape != (18, tam_ventana_mues):
            continue

        etiqueta = 0
        for inicio_a, fin_a in intervalos_ataque:
            if inicio_v < fin_a and fin_v > inicio_a:
                etiqueta = 1
                break

        coefs = extraer_coeficientes_dwt(ventana)
        X_lista.append(coefs)
        y_lista.append(etiqueta)

    del raw, senal

    if len(X_lista) == 0:
        return None, None, None

    X = np.asarray(X_lista, dtype=np.float32)
    y = np.asarray(y_lista, dtype=np.int32)
    return X, y, nombre_edf


# ==========================================
# 3. Procesamiento principal
# ==========================================
print("Iniciando procesamiento CHB-MIT con DWT (18 canales) -> BIN...\n")
resumen_global = []

index_rows = []
feature_dim = None
global_idx = 0

with open(bin_signals_path, "wb") as f_signals, open(bin_labels_path, "wb") as f_labels:

    for paciente in tqdm(pacientes, desc="Pacientes", position=0):
        ruta_paciente = os.path.join(ruta_dataset, paciente)

        if not os.path.exists(ruta_paciente):
            tqdm.write(f"Advertencia: No se encontró '{paciente}'. Omitiendo...")
            continue

        ruta_summary = os.path.join(ruta_paciente, f"{paciente}-summary.txt")
        if not os.path.exists(ruta_summary):
            tqdm.write(f"Advertencia: No se encontró summary de '{paciente}'. Omitiendo...")
            continue

        ataques_por_archivo = parsear_summary(ruta_summary)
        archivos_edf = sorted([f for f in os.listdir(ruta_paciente) if f.lower().endswith(".edf")])

        total_ventanas = 0
        total_ataques = 0
        archivos_ok = 0
        archivos_omitidos = 0

        for nombre_edf in tqdm(archivos_edf, desc=f"{paciente}", position=1, leave=False):
            ruta_edf = os.path.join(ruta_paciente, nombre_edf)
            nombre_base = os.path.splitext(nombre_edf)[0]

            try:
                X, y, _ = procesar_edf(ruta_edf, nombre_edf, ataques_por_archivo)

                if X is None or y is None:
                    tqdm.write(f"Omitido (no cumple 18 canales válidos o sin ventanas): {nombre_edf}")
                    archivos_omitidos += 1
                    continue

                if feature_dim is None:
                    feature_dim = X.shape[1]

                if X.shape[1] != feature_dim:
                    tqdm.write(f"Omitido (feature_dim inconsistente): {nombre_edf}")
                    archivos_omitidos += 1
                    continue

                X.astype(np.float32).tofile(f_signals)
                y.astype(np.float32).tofile(f_labels)

                for i in range(len(y)):
                    index_rows.append([
                        global_idx,
                        paciente,
                        nombre_base,
                        i,
                        int(y[i]),
                    ])
                    global_idx += 1

                total_ventanas += len(y)
                total_ataques += int(np.sum(y == 1))
                archivos_ok += 1

                tqdm.write(f"{nombre_edf} -> Shape: {X.shape} | Ataques: {int(np.sum(y == 1))}")

                del X, y

            except Exception as e:
                tqdm.write(f"ERROR en {nombre_edf}: {e}")
                archivos_omitidos += 1

        resumen_global.append((paciente, total_ventanas, total_ataques, archivos_ok, archivos_omitidos))
        tqdm.write(
            f"\n{paciente} completado -> Ventanas: {total_ventanas} | "
            f"Con ataque: {total_ataques} | EDF OK: {archivos_ok} | EDF omitidos: {archivos_omitidos}\n"
        )

with open(index_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["global_idx", "patient_id", "record_id", "window_idx", "label"])
    writer.writerows(index_rows)

print("\n" + "=" * 60)
print("RESUMEN GLOBAL")
print("=" * 60)

total_v, total_a, total_ok, total_omit = 0, 0, 0, 0
for paciente, ventanas, ataques, ok, omitidos in resumen_global:
    print(
        f"{paciente} -> Ventanas: {ventanas} | Con ataque: {ataques} | "
        f"EDF OK: {ok} | EDF omitidos: {omitidos}"
    )
    total_v += ventanas
    total_a += ataques
    total_ok += ok
    total_omit += omitidos

print(f"\nTotal ventanas   : {total_v}")
print(f"Total ataques    : {total_a}")
print(f"Total EDF OK     : {total_ok}")
print(f"Total omitidos   : {total_omit}")
print(f"Archivos en      : {carpeta_salida}")
print(f"Signals bin      : {bin_signals_path}")
print(f"Labels bin       : {bin_labels_path}")
print(f"Index csv        : {index_csv_path}")
print(f"Canales usados   : {len(SOTA_CHANNELS)}")
print(f"Wavelet / nivel  : {wavelet} / {nivel_descomp}")
print(f"Feature dim      : {feature_dim}")