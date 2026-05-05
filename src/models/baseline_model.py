import torch
import torch.nn as nn

class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()
        
        # --- BLOQUE 1: Extracción Temporal ---
        self.block1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(64, 1)

    def forward(self, x):
        
        """
        x shape: [Batch, Canales, 256]
        """
        batch_size, num_channels, _ = x.shape

        # Channel-independent
        x = x.view(-1, 1, 256)        # [B*C, 1, 256]

        x = self.block1(x)            # [B*C, 16, 128]
        x = self.block2(x)            # [B*C, 32, 64]
        x = self.block3(x)            # [B*C, 64, 64]

        x = self.global_pool(x)       # [B*C, 64, 1]
        x = x.squeeze(-1)             # [B*C, 64]

        logits = self.classifier(x)   # [B*C, 1]
        logits = logits.squeeze(-1)   # [B*C]

        # Voting por paciente
        logits = logits.view(batch_size, num_channels)
        logits = logits.mean(dim=1)   # [Batch]

        return logits



