import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class DWTSeizureDataset(Dataset):
    def __init__(
        self,
        signals_path,
        labels_path,
        index_csv_path,
        indices=None,
        num_samples=None,
        feature_dim=None,
        normalize=True,
    ):
        self.signals_path = str(signals_path)
        self.labels_path = str(labels_path)
        self.index_csv_path = str(index_csv_path)
        self.normalize = normalize

        self.index_table = []
        with open(self.index_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.index_table.append(row)

        if indices is None:
            self.indices = np.arange(len(self.index_table))
        else:
            self.indices = np.asarray(indices)

        if num_samples is None:
            num_samples = len(self.index_table)
        if feature_dim is None:
            raise ValueError("feature_dim es obligatorio para abrir el memmap.")

        self.num_samples = int(num_samples)
        self.feature_dim = int(feature_dim)

        self.signals = None
        self.labels = None

    def __len__(self):
        return len(self.indices)

    def _lazy_init(self):
        if self.signals is None:
            self.signals = np.memmap(
                self.signals_path,
                dtype="float32",
                mode="r",
                shape=(self.num_samples, self.feature_dim),
            )
            self.labels = np.memmap(
                self.labels_path,
                dtype="float32",
                mode="r",
                shape=(self.num_samples,),
            )

    def __getitem__(self, idx):
        self._lazy_init()

        real_idx = int(self.indices[idx])
        x = np.array(self.signals[real_idx], dtype=np.float32, copy=True)
        y = np.float32(self.labels[real_idx])

        if self.normalize:
            mean = x.mean()
            std = x.std()
            x = (x - mean) / (std + 1e-6)

        x = torch.from_numpy(x)
        y = torch.tensor(y, dtype=torch.float32)

        meta = self.index_table[real_idx]
        record_id = meta["record_id"]
        patient_id = meta["patient_id"]

        return x, y, record_id, patient_id