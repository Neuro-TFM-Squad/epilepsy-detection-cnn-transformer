import os
import torch
import pandas as pd
from torch.utils.data import Dataset

class EEGTransforms:
    """
    Clase para aplicar transformaciones estocásticas a señales EEG.
    Se utiliza PyTorch para mantener las operaciones eficientes en tensores.
    """
    def __init__(self, noise_std_range=(0.01, 0.05), scale_range=(0.8, 1.2)):
        self.noise_min, self.noise_max = noise_std_range
        self.scale_min, self.scale_max = scale_range

    def __call__(self, signal):
        # 1. Random Amplitude Scale (Variación de ganancia)
        # Generamos un factor escalar aleatorio en una distribución uniforme
        scale_factor = torch.empty(1).uniform_(self.scale_min, self.scale_max).item()
        signal = signal * scale_factor

        # 2. Gaussian Noise (Ruido blanco)
        # Generamos una desviación estándar aleatoria para el ruido
        std = torch.empty(1).uniform_(self.noise_min, self.noise_max).item()
        # Creamos el ruido con la misma forma que la señal [C, 256]
        noise = torch.randn_like(signal) * std
        signal = signal + noise

        return signal


class SeizureDataset(Dataset):
    def __init__(self, csv_file, root_dir, expected_channels=18, augment=False):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.expected_channels = expected_channels
        self.augment = augment
        
        # Instanciar las transformaciones en el constructor por eficiencia
        self.transforms = EEGTransforms() if self.augment else None

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        file_name = self.annotations.iloc[idx]['filepath']
        label = int(self.annotations.iloc[idx]['label'])
        tensor_path = os.path.join(self.root_dir, file_name)
        
        # weights_only=True es una buena práctica de seguridad en PyTorch
        data_dict = torch.load(tensor_path, weights_only=True)
        signal = data_dict['signal'] # Forma actual: [C, 256]
        
        # Aplicar transformaciones si el flag está activo
        if self.augment and self.transforms:
            signal = self.transforms(signal)
        
        return signal, torch.tensor(label, dtype=torch.long)