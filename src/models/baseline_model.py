import torch
import torch.nn as nn

class BaselineCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=1): 
        super(BaselineCNN, self).__init__()
        
        # --- BLOQUE 1: Extracción Temporal ---
        self.block1 = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=16,
                kernel_size=15,
                padding=7
            ),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                padding=2
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
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



