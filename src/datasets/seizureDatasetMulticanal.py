import torch
import numpy as np
<<<<<<< HEAD
import h5py
from torch.utils.data import Dataset

class EEGTransforms:
    """Mismas transformaciones estocásticas que usabas en Bonn.
    Funcionan perfectamente tanto para [C, L] como para secuencias [S, C, L]"""
=======
from torch.utils.data import Dataset

class EEGTransforms:
>>>>>>> master
    def __init__(self, noise_std_range=(0.01, 0.05), scale_range=(0.8, 1.2)):
        self.noise_min, self.noise_max = noise_std_range
        self.scale_min, self.scale_max = scale_range

    def __call__(self, signal):
        scale_factor = torch.empty(1).uniform_(self.scale_min, self.scale_max).item()
        signal = signal * scale_factor
        std = torch.empty(1).uniform_(self.noise_min, self.noise_max).item()
        noise = torch.randn_like(signal) * std
        return signal + noise

<<<<<<< HEAD

class SeizureDatasetMultichannel(Dataset):
    def __init__(self, h5_path, augment=False, sequence_length=1):
        """
        Args:
            h5_path (str): Ruta al archivo HDF5.
            augment (bool): Si aplica data augmentation.
            sequence_length (int): Número de ventanas seguidas a devolver. 
                                   1 para CNN estática, >1 para LSTM/Transformers.
        """
        self.h5_path = h5_path
=======
class SeizureDatasetMultichannel(Dataset):
    def __init__(self, signals_path, labels_path, augment=False, sequence_length=1):
        self.signals_path = signals_path
        self.labels_path = labels_path
        
>>>>>>> master
        self.augment = augment
        self.sequence_length = sequence_length
        self.transforms = EEGTransforms() if self.augment else None
        
<<<<<<< HEAD
        # Leemos metadatos básicos
        with h5py.File(self.h5_path, 'r') as f:
            self.num_samples = len(f['labels'])
            self.labels = f['labels'][:]  
            
        self.h5_file = None
=======
        # ⚠️ EL NÚMERO MÁGICO ⚠️
        # Pon aquí el número exacto que te dio el script anterior por consola
        self.num_samples = 1352243 
        
        # Dejamos esto en None para que los workers no exploten la RAM
        self.signals = None
        self.labels = None
>>>>>>> master

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
<<<<<<< HEAD
        # Inicialización lazy para workers
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')
            
        # =========================================================
        # LÓGICA DE EXTRACCIÓN: MONO-VENTANA VS SECUENCIA
        # =========================================================
        if self.sequence_length == 1:
            # Caso Base (Para tu CNN Espacio-Temporal actual)
            # Extraemos: [18 canales, 256 muestras]
            signal = self.h5_file['signals'][idx]
            label = self.labels[idx]
            
        else:
            # Caso Avanzado (Para futura LSTM/Transformer)
            # Extraemos una secuencia causal: [idx - seq_len + 1 : idx + 1]
            # Ej: Si idx=10 y seq=5, coge de la ventana 6 a la 10.
            start_idx = max(0, idx - self.sequence_length + 1)
            signal_seq = self.h5_file['signals'][start_idx : idx + 1]
            
            # --- Padding para los primeros segundos del dataset ---
            # Si idx=2 y seq=5, solo tenemos 3 ventanas reales. Rellenamos copiando la primera.
            pad_len = self.sequence_length - len(signal_seq)
            if pad_len > 0:
                pad = np.repeat(signal_seq[0:1], pad_len, axis=0) # Copiamos la primera ventana
                signal_seq = np.concatenate([pad, signal_seq], axis=0)
                
            signal = signal_seq # Shape resultante: [S, 18, 256]
            label = self.labels[idx] # La etiqueta siempre es la del instante ACTUAL (idx)

        # =========================================================
        # CONVERSIÓN Y AUMENTO
        # =========================================================
        signal = torch.tensor(signal, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long) # O torch.float32 si usas BCEWithLogitsLoss
=======
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
>>>>>>> master
        
        if self.augment and self.transforms:
            signal = self.transforms(signal)
            
        return signal, label