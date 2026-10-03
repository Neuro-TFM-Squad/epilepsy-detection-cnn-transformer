# Resolve the repository for both module and direct script execution.
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import os
import ast
import mne
import numpy as np
import pandas as pd
from tqdm import tqdm


SOTA_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ"
]


def _parse_seizure_times(value):
    if pd.isna(value):
        return []

    s = str(value).strip()
    if s in ("", "[]", "nan", "None"):
        return []

    try:
        parsed = ast.literal_eval(s)
        if isinstance(parsed, tuple):
            parsed = [parsed]

        events = []
        for ev in parsed:
            if isinstance(ev, (list, tuple)) and len(ev) == 2:
                start, end = float(ev[0]), float(ev[1])
                if end < start:
                    start, end = end, start
                events.append((start, end))
        return events
    except Exception:
        return []


def _harmonize_channels(raw, target_channels):
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


def process_chbmit_to_bin(
    raw_dir,
    output_dir,
    csv_path,
    target_fs=256,
    window_sec=1,
    stride_sec=1
):
    os.makedirs(output_dir, exist_ok=True)

    if window_sec <= 0 or stride_sec <= 0:
        raise ValueError("window_sec y stride_sec deben ser > 0.")

    if target_fs <= 0:
        raise ValueError("target_fs debe ser > 0.")

    window_size = int(window_sec * target_fs)
    stride_size = int(stride_sec * target_fs)

    if window_size != target_fs:
        print(f"⚠️ Aviso: con target_fs={target_fs}, una ventana de {window_sec}s equivale a {window_size} muestras.")

    signals_bin_path = os.path.join(output_dir, "chbmit_signals.bin")
    labels_bin_path = os.path.join(output_dir, "chbmit_labels.bin")
    index_csv_path = os.path.join(output_dir, "chbmit_index.csv")

    metadata_list = []
    total_windows = 0
    total_positive = 0
    skipped_missing_file = 0
    skipped_read_error = 0
    skipped_missing_channels = 0
    skipped_too_short = 0

    df_labels = pd.read_csv(csv_path)

    with open(signals_bin_path, "wb") as f_sig, open(labels_bin_path, "wb") as f_lab:
        for _, row in tqdm(df_labels.iterrows(), total=len(df_labels), desc="Procesando EDFs"):
            patient_id = row["paciente"]
            edf_file = row["archivo"]
            seizure_times = _parse_seizure_times(row["crisis"])

            edf_path = os.path.join(raw_dir, patient_id, edf_file)
            if not os.path.exists(edf_path):
                print(f"⚠️ No existe: {edf_path}")
                skipped_missing_file += 1
                continue

            try:
                raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
            except Exception as e:
                print(f"❌ Error leyendo {edf_file}: {e}")
                skipped_read_error += 1
                continue

            try:
                raw = _harmonize_channels(raw, SOTA_CHANNELS)
            except Exception as e:
                print(f"⚠️ {edf_file} no tiene los 18 canales exactos. Saltando... Error: {e}")
                skipped_missing_channels += 1
                continue

            if int(raw.info["sfreq"]) != target_fs:
                raw.resample(target_fs, npad="auto")

            data = raw.get_data().astype(np.float32)  # [18, T]
            total_samples = data.shape[1]

            if total_samples < window_size:
                print(f"⚠️ {edf_file} demasiado corto para ventana de {window_sec}s. Saltando.")
                skipped_too_short += 1
                continue

            n_windows = 1 + (total_samples - window_size) // stride_size
            record_id = os.path.splitext(edf_file)[0]

            for win_idx in range(n_windows):
                start_idx = win_idx * stride_size
                end_idx = start_idx + window_size

                window = data[:, start_idx:end_idx]
                if window.shape != (18, window_size):
                    continue

                window_start_sec = start_idx / target_fs
                window_end_sec = end_idx / target_fs

                label = 0
                for seiz_start, seiz_end in seizure_times:
                    if window_start_sec < seiz_end and window_end_sec > seiz_start:
                        label = 1
                        break

                f_sig.write(window.tobytes())
                f_lab.write(np.array([label], dtype=np.float32).tobytes())

                metadata_list.append({
                    "patient_id": patient_id,
                    "record_id": record_id,
                    "filepath": edf_file,
                    "window_index": win_idx,
                    "start_sample": start_idx,
                    "end_sample": end_idx,
                    "start_sec": window_start_sec,
                    "end_sec": window_end_sec,
                    "window_sec": float(window_sec),
                    "stride_sec": float(stride_sec),
                    "label": int(label)
                })

                total_windows += 1
                total_positive += label

    df_out = pd.DataFrame(metadata_list)
    df_out.to_csv(index_csv_path, index=False)

    print("\n✅ PREPROCESADO COMPLETADO")
    print(f"📦 Señales BIN: {signals_bin_path}")
    print(f"🏷️ Labels BIN: {labels_bin_path}")
    print(f"📄 Índice CSV: {index_csv_path}")
    print(f"🔢 Total ventanas: {total_windows}")
    print(f"🚨 Ventanas positivas: {total_positive}")
    print(f"🌑 Ventanas negativas: {total_windows - total_positive}")

    if total_windows > 0:
        print(f"⚖️ Ratio neg/pos: {(total_windows - total_positive) / max(total_positive, 1):.2f}")

    print(f"📐 Forma esperada por ventana: (18, {window_size})")
    print(f"⏭️ EDF ausentes: {skipped_missing_file}")
    print(f"📖 EDF con error de lectura: {skipped_read_error}")
    print(f"🧠 EDF sin 18 canales válidos: {skipped_missing_channels}")
    print(f"⏱️ EDF demasiado cortos: {skipped_too_short}")


if __name__ == "__main__":
    process_chbmit_to_bin(
        raw_dir="data/CHBMIT/raw/",
        output_dir="data/CHBMIT/processed/ventana1s",
        csv_path="data/metadata/metadata_chbmit_etiquetado.csv",
        target_fs=256,
        window_sec=1,
        stride_sec=1
    )