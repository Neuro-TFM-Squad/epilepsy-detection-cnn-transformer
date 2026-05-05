import os
import torch
import pandas as pd
import h5py
from tqdm import tqdm

def create_hdf5_dataset():
    # Rutas de entrada
    csv_file = "data/CHB-MIT/processed/chbmit_index.csv"
    root_dir = "data/CHB-MIT/processed/"
    
    # Ruta de salida (el nuevo super-archivo)
    h5_output_path = "data/CHB-MIT/processed/chbmit_dataset.h5"
    
    print("Lectura de CSV...")
    df = pd.read_csv(csv_file)
    num_samples = len(df)
    
    # Abrimos el archivo HDF5 en modo escritura ('w')
    with h5py.File(h5_output_path, 'w') as h5f:
        # Creamos los contenedores (datasets) dentro del HDF5
        signals_dset = h5f.create_dataset("signals", shape=(num_samples, 18, 256), dtype='float32')
        labels_dset = h5f.create_dataset("labels", shape=(num_samples,), dtype='int8')
        
        print(f"Empaquetando {num_samples} archivos en HDF5. Esto tomará un rato, pero solo se hace una vez...")
        
        for i in tqdm(range(num_samples), desc="Convirtiendo a HDF5"):
            file_name = df.iloc[i]['filepath']
            label = int(df.iloc[i]['label'])
            
            tensor_path = os.path.join(root_dir, file_name)
            data_dict = torch.load(tensor_path, weights_only=True)
            signal_tensor = data_dict['signal'] 
            
            # Inyectar en el HDF5
            signals_dset[i] = signal_tensor.numpy()
            labels_dset[i] = label
            
    print(f"\n✅ ¡Éxito! Tu dataset completo y ultrarrápido está ahora en: {h5_output_path}")

if __name__ == "__main__":
    create_hdf5_dataset()