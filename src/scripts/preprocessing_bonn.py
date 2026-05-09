import os
import numpy as np
import torch
import pandas as pd
from scipy.signal import resample

def process_bonn(raw_dir, output_dir, original_fs=173.61, target_fs=256):
    """
    Procesa el dataset de Bonn.
    raw_dir: Ruta a la carpeta que contiene set_A, set_B, set_C, set_D, set_E.
    """
    os.makedirs(output_dir, exist_ok=True)
    metadata = []
    
    # Mapeo estricto: Solo set_E contiene actividad Ictal (Crisis = 1)
    folder_to_label = {'set_A': 0, 'set_B': 0, 'set_C': 0, 'set_D': 0, 'set_E': 1}
    
    for folder, label in folder_to_label.items():
        folder_path = os.path.join(raw_dir, folder)
        if not os.path.exists(folder_path):
            print(f"⚠️ Carpeta no encontrada: {folder_path}. Saltando...")
            continue
            
        print(f"Procesando {folder} (Etiqueta {label})...")
        
        for file in os.listdir(folder_path):
            if not file.endswith('.txt'):
                continue
                
            file_path = os.path.join(folder_path, file)
            
            # 1. Cargar señal cruda (Bonn tiene 4097 muestras por TXT)
            signal = np.loadtxt(file_path)
            
            # 2. Resampling al estándar del modelo (256 Hz)
            duration = len(signal) / original_fs
            target_length = int(duration * target_fs)
            signal_resampled = resample(signal, target_length)
            
            # 3. Normalización Z-Score
            mean = np.mean(signal_resampled)
            std = np.std(signal_resampled)
            signal_norm = (signal_resampled - mean) / (std + 1e-8)
            
            # 4. Ventanado estricto de 1 segundo (256 muestras)
            n_windows = len(signal_norm) // target_fs
            
            for i in range(n_windows):
                start_idx = i * target_fs
                end_idx = start_idx + target_fs
                window = signal_norm[start_idx:end_idx]
                
                # Crear tensor con dimensión explícita de canal: [1, 256]
                tensor_data = torch.tensor(window, dtype=torch.float32).unsqueeze(0)
                
                # Guardar tensor
                out_filename = f"bonn_{folder}_{file.replace('.txt', '')}_win{i:03d}.pt"
                torch.save({'signal': tensor_data}, os.path.join(output_dir, out_filename))
                
                # Registrar para el DataLoader
                metadata.append({'filepath': out_filename, 'label': label})
                
    # Guardar el índice maestro
    df = pd.DataFrame(metadata)
    csv_path = os.path.join(output_dir, 'bonn_index.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✅ BONN COMPLETADO: {len(df)} ventanas generadas. Índice en {csv_path}")

# Ejemplo de uso:
process_bonn(
    raw_dir='data/BONN/raw', 
    output_dir='data/BONN/processed'
)