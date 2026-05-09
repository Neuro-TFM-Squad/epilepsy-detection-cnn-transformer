import h5py
import numpy as np
from tqdm import tqdm

def main():
    h5_path = "data/CHB-MIT/processed/chbmit_dataset.h5"
    # Lo guardamos como .bin (binario puro) en lugar de .npy
    signals_bin = "data/CHB-MIT/processed/chbmit_signals.bin"
    labels_bin = "data/CHB-MIT/processed/chbmit_labels.bin"
    
    print("Abriendo HDF5...")
    with h5py.File(h5_path, 'r') as f:
        signals_h5 = f['signals']
        labels_h5 = f['labels']
        num_samples = len(labels_h5)
        
        print(f"Total de ventanas a convertir: {num_samples}")
        
        # Leemos en bloques pequeñitos y escribimos directamente al disco
        chunk_size = 5000 
        
        # 'wb' asegura que va directo al disco físico, sin cachés raras de RAM
        with open(signals_bin, 'wb') as f_sig, open(labels_bin, 'wb') as f_lab:
            for i in tqdm(range(0, num_samples, chunk_size), desc="Escribiendo Binario"):
                end = min(i + chunk_size, num_samples)
                
                # Leemos de HDF5 y convertimos a bytes en crudo
                sig_chunk = signals_h5[i:end].astype('float32').tobytes()
                # Aprovechamos para guardar las labels ya en float32 (te ahorrará un paso en PyTorch)
                lab_chunk = labels_h5[i:end].astype('float32').tobytes()
                
                f_sig.write(sig_chunk)
                f_lab.write(lab_chunk)
                
    print("✅ ¡Conversión binaria completada con éxito!")

if __name__ == "__main__":
    main()