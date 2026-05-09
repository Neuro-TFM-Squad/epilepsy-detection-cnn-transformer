import os
import mne
import pandas as pd
import numpy as np
import torch
import warnings
from tqdm import tqdm

# Silenciar las advertencias de canales duplicados de MNE
warnings.filterwarnings('ignore', category=RuntimeWarning)
mne.set_log_level('ERROR')

# --- CONFIGURACIÓN DE RUTAS ---
RAW_DATA_DIR = "data/CHBMIT/raw"
PROCESSED_DATA_DIR = "data/CHBMIT/processed"
INDEX_CSV = "data/CHBMIT/processed/chbmit_index.csv"

# Archivos binarios de salida (streaming directo al SSD) 
SIGNALS_BIN = os.path.join(PROCESSED_DATA_DIR, "subset_signals.bin")
LABELS_BIN = os.path.join(PROCESSED_DATA_DIR, "subset_labels.bin")

FS = 256
CHANNELS = [
    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 
    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 
    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 
    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 
    'FZ-CZ', 'CZ-PZ'
]

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

def preprocess_and_save_subset():
    df = pd.read_csv(INDEX_CSV)
    df['patient'] = df['filepath'].apply(lambda x: x.split('_')[0])
    df['file_id'] = df['filepath'].apply(lambda x: "_".join(x.split('_')[:2]))
    
    # 🔥 SOLO PROCESAMOS ESTOS 6 PACIENTES PARA EL TFM
    target_patients = ['chb01', 'chb02', 'chb03', 'chb04', 'chb05', 'chb06']
    df = df[df['patient'].isin(target_patients)]
    
    total_ventanas_guardadas = 0

    # Abrimos los archivos binarios en modo "append binary" (ab)
    # Esto escribe directamente en el disco duro SSD sin llenar la RAM
    with open(SIGNALS_BIN, 'wb') as f_sig, open(LABELS_BIN, 'wb') as f_lab:
        
        for patient in target_patients:
            patient_df = df[df['patient'] == patient]
            unique_files = patient_df['file_id'].unique()
            
            print(f"\n🧠 Procesando paciente {patient}...")
            
            for file_id in tqdm(unique_files, desc=f"Archivos de {patient}"):
                input_file = os.path.join(RAW_DATA_DIR, patient, f"{file_id}.edf")
                if not os.path.exists(input_file):
                    continue

                try:
                    # 1. CARGA Y LIMPIEZA
                    raw = mne.io.read_raw_edf(input_file, preload=True, verbose=False)
                    raw.rename_channels(lambda x: x.strip().replace('.', ''))
                    
                    mapping = {}
                    for ch in raw.ch_names:
                        if ch.endswith('-0') or ch.endswith('-1'):
                            base_name = ch[:-2]
                            if base_name in CHANNELS and base_name not in mapping.values():
                                mapping[ch] = base_name
                    raw.rename_channels(mapping)

                    actual_names = raw.ch_names
                    inversions = {}
                    for target in CHANNELS:
                        if target not in actual_names:
                            parts = target.split('-')
                            inverted = f"{parts[1]}-{parts[0]}"
                            if inverted in actual_names:
                                inversions[inverted] = target
                    raw.rename_channels(inversions)

                    existing_channels = [ch for ch in CHANNELS if ch in raw.ch_names]
                    if len(existing_channels) < 18:
                        continue
                    
                    raw.pick(CHANNELS) 
                    raw.filter(l_freq=0.5, h_freq=40, verbose=False)
                    raw.notch_filter(freqs=50, verbose=False)
                    data = raw.get_data()

                    # 2. PROCESADO POR VENTANAS Y ESCRITURA INMEDIATA
                    relevant_rows = patient_df[patient_df['file_id'] == file_id]
                    
                    for _, row in relevant_rows.iterrows():
                        target_filename = row['filepath']
                        win_idx = int(target_filename.split('win')[-1].replace('.pt', ''))
                        
                        start = win_idx * FS
                        end = start + FS
                        
                        if end > data.shape[1]:
                            continue
                        
                        segment = data[:, start:end]
                        
                        # Z-Score y conversión inmediata a float32 para ahorrar espacio
                        mean = np.mean(segment, axis=1, keepdims=True)
                        std = np.std(segment, axis=1, keepdims=True)
                        segment = ((segment - mean) / (std + 1e-8)).astype(np.float32)
                        label = np.array([row['label']], dtype=np.float32)
                        
                        # 3. ESCRIBIR EN DISCO (¡Adiós problemas de RAM!)
                        f_sig.write(segment.tobytes())
                        f_lab.write(label.tobytes())
                        
                        total_ventanas_guardadas += 1

                    # Liberar memoria de este EDF
                    del raw
                    del data
                    
                except Exception as e:
                    pass # Ignoramos archivos corruptos silenciosamente
                    
    print("\n" + "="*50)
    print("✅ ¡PROCESO COMPLETADO CON ÉXITO!")
    print(f"⚠️ NÚMERO MÁGICO PARA EL DATASET: {total_ventanas_guardadas} ⚠️")
    print("Apunta este número para ponerlo en self.num_samples")
    print("="*50)

if __name__ == "__main__":
    preprocess_and_save_subset()