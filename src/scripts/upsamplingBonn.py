import numpy as np
import os
import glob
from scipy.signal import resample

def preprocess_and_save_bonn(raw_path, processed_path, target_len=1024):
    """
    Lee de 'raw', resamplea y guarda en 'processed'.
    """
    X, y = [], []
    folders = {
        'SET_A': 0, 'SET_B': 0, 
        'SET_C': 0, 'SET_D': 0, 
        'SET_E': 1
    }

    if not os.path.exists(processed_path):
        os.makedirs(processed_path)

    print(f"Leyendo de: {raw_path}")
    
    for folder, label in folders.items():
        path = os.path.join(raw_path, folder, "*.txt")
        files = glob.glob(path)
        print(f"Procesando {folder}: {len(files)} archivos...")
        
        for f in files:
            signal = np.loadtxt(f)
            # Sacamos 4 ventanas de 700 pts (aprox 4s a 173Hz)
            for i in range(4):
                start = i * 700
                end = start + 700
                if end <= len(signal):
                    window = signal[start:end]
                    # Resampling a 1024 (formato CHB-MIT)
                    window_resampled = resample(window, target_len)
                    X.append(window_resampled)
                    y.append(label)
                
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    # Guardar en formato binario para carga rápida
    np.save(os.path.join(processed_path, 'X_bonn_256Hz.npy'), X)
    np.save(os.path.join(processed_path, 'y_bonn_256Hz.npy'), y)
    
    print(f"¡Hecho! Archivos guardados en {processed_path}")
    print(f"Dimensiones finales: X={X.shape}, y={y.shape}")

# Ejecución manual del preprocesado
if __name__ == "__main__":
    RAW_DIR = r"D:\TFM\epilepsy-detection-cnn-transformer\data\BONN\raw"
    PROCESSED_DIR = r"D:\TFM\epilepsy-detection-cnn-transformer\data\BONN\processed"
    preprocess_and_save_bonn(RAW_DIR, PROCESSED_DIR)