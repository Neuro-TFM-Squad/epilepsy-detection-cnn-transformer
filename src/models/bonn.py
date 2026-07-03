import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.block2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.block3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.classifier(x)
        return x