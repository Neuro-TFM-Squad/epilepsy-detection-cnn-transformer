import torch
import numpy as np
from torch.utils.data import Dataset

class EEGTransforms:
    def __init__(self, noise_std_range=(0.01, 0.05), scale_range=(0.8, 1.2)):
        self.noise_min, self.noise_max = noise_std_range
        self.scale_min, self.scale_max = scale_range

    def __call__(self, signal):
        scale_factor = torch.empty(1).uniform_(self.scale_min, self.scale_max).item()
        signal = signal * scale_factor
        std = torch.empty(1).uniform_(self.noise_min, self.noise_max).item()
        noise = torch.randn_like(signal) * std
        return signal + noise

class SeizureDatasetMultichannel(Dataset):
    def __init__(self, signals_path, labels_path, augment=False, sequence_length=1):
        self.signals_path = signals_path
        self.labels_path = labels_path
        
        self.augment = augment
        self.sequence_length = sequence_length
        self.transforms = EEGTransforms() if self.augment else None
        
        # ⚠️ EL NÚMERO MÁGICO ⚠️
        # Pon aquí el número exacto que te dio el script anterior por consola
        self.num_samples = 1352243 
        
        # Dejamos esto en None para que los workers no exploten la RAM
        self.signals = None
        self.labels = None

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Inicialización Lazy para Windows
        if self.signals is None:
            self.signals = np.memmap(self.signals_path, dtype='float32', mode='r', shape=(self.num_samples, 18, 256))
            self.labels = np.memmap(self.labels_path, dtype='float32', mode='r', shape=(self.num_samples,))
            
        if self.sequence_length == 1:
            signal = self.signals[idx]
            label = self.labels[idx]
        else:
            start_idx = max(0, idx - self.sequence_length + 1)
            signal_seq = self.signals[start_idx : idx + 1]
            
            pad_len = self.sequence_length - len(signal_seq)
            if pad_len > 0:
                pad = np.repeat(signal_seq[0:1], pad_len, axis=0) 
                signal_seq = np.concatenate([pad, signal_seq], axis=0)
                
            signal = signal_seq 
            label = self.labels[idx] 

        # Conversión a tensores
        signal = torch.tensor(np.copy(signal), dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32) 
        
        if self.augment and self.transforms:
            signal = self.transforms(signal)
            
        return signal, label