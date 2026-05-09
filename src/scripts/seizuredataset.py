import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os

class SeizureDataset(Dataset):
    def __init__(self, csv_file, root_dir, expected_channels=18):
        """
        Args:
            csv_file (string): Ruta al index.csv generado en el preprocesado.
            root_dir (string): Carpeta donde están los .pt.
            expected_channels (int): 18 para cumplir con el estándar CHB-MIT.
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.expected_channels = expected_channels

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        file_name = self.annotations.iloc[idx]['filepath']
        label = int(self.annotations.iloc[idx]['label'])
        
        tensor_path = os.path.join(self.root_dir, file_name)
        data_dict = torch.load(tensor_path, weights_only=True)
        signal = data_dict['signal'] # Forma: [C, 256]
        
        c, t = signal.shape
        
        # ELIMINAMOS EL .expand() DE BONN. Dejamos que sea [1, 256].
        
        # Solo dejamos un filtro de seguridad por si algún archivo de CHB-MIT está corrupto
        if c != 1 and c != self.expected_channels:
            raise ValueError(f"Tensor {file_name} tiene {c} canales. Se esperaba 1 o {self.expected_channels}")
            
        return signal, torch.tensor(label, dtype=torch.long)

# ==========================================
# SANITY CHECK DEL DATALOADER
# ==========================================
if __name__ == "__main__":
    # Ajusta estas rutas a donde tengas tu carpeta de Bonn
    CSV_PATH_CHB = "data/CHB-MIT/processed/chbmit_index.csv"
    ROOT_DIR_CHB = "data/CHB-MIT/processed/"
    
    try:
        # Instanciamos el dataset de Bonn
        chbmit_dataset = SeizureDataset(csv_file=CSV_PATH_CHB, root_dir=ROOT_DIR_CHB)
        
        # Creamos el DataLoader (Pedimos un batch de 32 ventanas)
        train_loader = DataLoader(chbmit_dataset, batch_size=32, shuffle=True)
        
        # Extraemos el primer batch
        batch_x, batch_y = next(iter(train_loader))
        
        print("✅ DataLoader funcionando correctamente.")
        print(f"Forma del Batch X (Señales): {batch_x.shape}")
        print(f"Forma del Batch Y (Etiquetas): {batch_y.shape}")
        
    except Exception as e:
        print(f"❌ Error al cargar los datos: {e}")