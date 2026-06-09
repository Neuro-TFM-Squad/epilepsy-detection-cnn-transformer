import torch
import torch.nn as nn

class BaselineCNN(nn.Module):
<<<<<<< HEAD
    def __init__(self):
=======
    def __init__(self, in_channels=1, num_classes=1): 
>>>>>>> master
        super(BaselineCNN, self).__init__()
        
        # --- BLOQUE 1: Extracción Temporal ---
        self.block1 = nn.Sequential(
            nn.Conv1d(
<<<<<<< HEAD
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
=======
                in_channels=in_channels,
                out_channels=16,
                kernel_size=15,
                padding=7
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
>>>>>>> master
        )
        
        self.block2 = nn.Sequential(
            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
<<<<<<< HEAD
                stride=1,
=======
>>>>>>> master
                padding=2
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
<<<<<<< HEAD
            nn.MaxPool1d(kernel_size=2)
=======
            nn.MaxPool1d(kernel_size=2, stride=2)
>>>>>>> master
        )
        
        self.block3 = nn.Sequential(
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
<<<<<<< HEAD
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
=======
                padding=1
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Linear(64*(256//8), num_classes)

    def forward(self, x):
        # x shape: (Batch, 1, 256)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        x = x.view(x.size(0), -1) # Flatten
        x = self.classifier(x)
        
        return x
>>>>>>> master



