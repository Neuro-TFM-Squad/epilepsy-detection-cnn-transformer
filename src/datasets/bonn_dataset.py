import pandas as pd
import torch
from torch.utils.data import Dataset


class BonnEEGDataset(Dataset):
    def __init__(self, index_csv, root_dir, augment=False):
        self.df = pd.read_csv(index_csv).reset_index(drop=True)
        self.root_dir = root_dir
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row["filepath"] if "filepath" in row else row["file_path"]
        label = row["label"]

        full_path = filepath if filepath.startswith(self.root_dir) else f"{self.root_dir}/{filepath}"
        signal = torch.load(full_path, map_location="cpu")

        if isinstance(signal, dict):
            if "signal" in signal:
                signal = signal["signal"]
            elif "x" in signal:
                signal = signal["x"]
            else:
                raise KeyError(f"No encuentro la señal en el .pt: {full_path}")

        signal = signal.float()
        if signal.ndim == 1:
            signal = signal.unsqueeze(0)

        mean = signal.mean(dim=-1, keepdim=True)
        std = signal.std(dim=-1, keepdim=True)
        signal = (signal - mean) / (std + 1e-6)

        if self.augment:
            scale = torch.empty(1).uniform_(0.9, 1.1).item()
            noise = torch.randn_like(signal) * torch.empty(1).uniform_(0.0, 0.02).item()
            signal = signal * scale + noise

        return signal, torch.tensor(float(label), dtype=torch.float32)