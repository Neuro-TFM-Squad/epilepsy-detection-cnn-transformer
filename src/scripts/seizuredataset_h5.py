import torch
import h5py
from torch.utils.data import Dataset

class EEGTransforms:
    """Mismas transformaciones estocásticas que usabas en Bonn"""
    def __init__(self, noise_std_range=(0.01, 0.05), scale_range=(0.8, 1.2)):
        self.noise_min, self.noise_max = noise_std_range
        self.scale_min, self.scale_max = scale_range

    def __call__(self, signal):
        scale_factor = torch.empty(1).uniform_(self.scale_min, self.scale_max).item()
        signal = signal * scale_factor
        std = torch.empty(1).uniform_(self.noise_min, self.noise_max).item()
        noise = torch.randn_like(signal) * std
        return signal + noise

class SeizureDatasetH5(Dataset):
    def __init__(self, h5_path, augment=False):
        self.h5_path = h5_path
        self.augment = augment
        self.transforms = EEGTransforms() if self.augment else None
        
        # Truco multiproceso: Leemos la longitud y las etiquetas, luego cerramos
        with h5py.File(self.h5_path, 'r') as f:
            self.num_samples = len(f['labels'])
            self.labels = f['labels'][:]  # <--- AÑADE ESTA LÍNEA
            
        self.h5_file = None

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Cada worker de PyTorch abrirá el archivo la primera vez que pida un dato
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
            
        # Extraemos directamente la señal y la etiqueta (es rapidísimo)
        # HDF5 devuelve arrays de numpy, los pasamos a tensores
        signal = torch.tensor(self.h5_file['signals'][idx]) # [18, 256]
        label = torch.tensor(self.h5_file['labels'][idx], dtype=torch.long)
        
        if self.augment and self.transforms:
            signal = self.transforms(signal)
            
        return signal, label