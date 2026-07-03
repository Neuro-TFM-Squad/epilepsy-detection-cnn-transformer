import torch
import torch.nn as nn

class DWTMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 128), dropout=0.25):
        super().__init__()
        dims = [input_dim] + list(hidden_dims)

        layers = []
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.BatchNorm1d(dims[i + 1]),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])

        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)