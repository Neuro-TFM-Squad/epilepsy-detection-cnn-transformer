import os
import mne
import numpy as np
import torch
import pandas as pd
import ast

def process_chbmit_with_csv(raw_dir, output_dir, csv_path, target_fs=256):
    os.makedirs(output_dir, exist_ok=True)
    metadata_list = []
    
    # Los 18 canales estándar (SOTA) que garantizan que no haya "Domain Shift" interno
    SOTA_CHANNELS = [
        'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1', 
        'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 
        'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 
        'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2', 
        'FZ-CZ', 'CZ-PZ'
    ]
    
    df_labels = pd.read_csv(csv_path)
    
    for index, row in df_labels.iterrows():
        patient_id = row['paciente']
        edf_file = row['archivo']
        
        if pd.isna(row['crisis']):
            seizure_times = []
        else:
            seizure_times = ast.literal_eval(row['crisis'])
            
        edf_path = os.path.join(raw_dir, patient_id, edf_file)
        if not os.path.exists(edf_path):
            continue
            
        print(f"Procesando {edf_file}...")
        
        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose='ERROR')
        except Exception as e:
            print(f"❌ Error leyendo {edf_file}: {e}")
            continue
        
        rename_dict = {}
        channels_to_drop = []
        
        # 1. Identificar qué canales "limpios" ya existen nativamente en el archivo
        existing_clean_channels = set(raw.ch_names).intersection(set(SOTA_CHANNELS))
        
        for ch in raw.ch_names:
            if ch.endswith('-0') or ch.endswith('-1'):
                clean_name = ch[:-2]
                
                if clean_name in SOTA_CHANNELS:
                    # Si ya tenemos el canal limpio, o ya programamos renombrar a un clon previo...
                    # Este canal es un duplicado redundante. ¡A la basura!
                    if clean_name in existing_clean_channels or clean_name in rename_dict.values():
                        channels_to_drop.append(ch)
                    else:
                        # Si es la primera vez que lo vemos, nos lo quedamos y lo renombramos
                        rename_dict[ch] = clean_name
        
        # 2. PRIMERO borramos los clones para que MNE no se queje de nombres duplicados
        if channels_to_drop:
            raw.drop_channels(channels_to_drop)
            
        # 3. LUEGO renombramos el único superviviente de forma segura
        if rename_dict:
            raw.rename_channels(rename_dict)
            
        # 2. Filtrado estricto de los 18 canales (Ignoramos el resto)
        try:
            # Hay un bug conocido en MNE con CHB-MIT: a veces T8-P8 se lee como T8-P8-1
            # Para simplificar, forzamos la extracción exacta.
            raw.pick_channels(SOTA_CHANNELS)
            raw.reorder_channels(SOTA_CHANNELS) # Garantiza el mismo orden espacial siempre
        except ValueError as e:
            print(f"⚠️ {edf_file} no tiene los 18 canales exactos. Saltando... Error: {e}")
            continue
            
        data = raw.get_data() # Ahora data tiene garantizada la forma [18, Tiempo]
        
        # 3. Normalización Z-Score Intracanal
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True)
        data = (data - mean) / (std + 1e-8)
        
        # 4. Ventanado (1 segundo)
        n_windows = data.shape[1] // target_fs
        
        for current_sec in range(n_windows):
            label = 0
            for (start, end) in seizure_times:
                if start <= current_sec <= end:
                    label = 1
                    break
                    
            start_idx = current_sec * target_fs
            end_idx = start_idx + target_fs
            window = data[:, start_idx:end_idx] # Forma asegurada: [18, 256]
            
            if window.shape[0] != 18 or window.shape[1] != target_fs:
                continue 
            
            tensor_data = torch.tensor(window, dtype=torch.float32)
            out_filename = f"{edf_file.replace('.edf', '')}_win{current_sec:04d}.pt"
            
            torch.save({'signal': tensor_data}, os.path.join(output_dir, out_filename))
            metadata_list.append({'filepath': out_filename, 'label': label})
            
    df_out = pd.DataFrame(metadata_list)
    out_csv_path = os.path.join(output_dir, 'chbmit_index.csv')
    df_out.to_csv(out_csv_path, index=False)
    print(f"\n✅ COMPLETADO: {len(df_out)} ventanas guardadas con forma [18, 256].")

process_chbmit_with_csv(
raw_dir='data/CHB-MIT/raw/', 
output_dir='data/CHB-MIT/processed/', 
csv_path='src/utils/metadata_chbmit_etiquetado.csv'
)