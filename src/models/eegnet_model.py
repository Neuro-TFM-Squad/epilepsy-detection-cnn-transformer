import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGNet(nn.Module):
    def __init__(self, in_channels=18, window_size=256, num_classes=1, F1=8, D=2, F2=16):
        super(EEGNet, self).__init__()
        
        self.in_channels = in_channels
        self.window_size = window_size
        
        # ==========================================
        # BLOQUE 1: Convolución Temporal
        # Extrae frecuencias de banda del EEG
        # ==========================================
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.batchnorm1 = nn.InstanceNorm2d(F1)
        
        # ==========================================
        # BLOQUE 2: Convolución Espacial (Depthwise)
        # Aprende qué electrodos son importantes sin mezclarlos todos a lo bruto
        # ==========================================
        self.depthwise1 = nn.Conv2d(F1, F1 * D, (in_channels, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.InstanceNorm2d(F1 * D)
        self.pooling1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(p=0.25)
        
        # ==========================================
        # BLOQUE 3: Convolución Separable
        # Combina la información temporal y espacial de forma súper eficiente
        # ==========================================
        self.separable_depth = nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False)
        self.separable_point = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.batchnorm3 = nn.InstanceNorm2d(F2)
        self.pooling2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(p=0.25)
        
        # ==========================================
        # CÁLCULO DINÁMICO DE LA DIMENSIÓN FINAL
        # ==========================================
        dummy = torch.zeros(1, 1, in_channels, window_size)
        with torch.no_grad():
            x = self.conv1(dummy)
            x = self.batchnorm1(x)
            x = self.depthwise1(x)
            x = self.batchnorm2(x)
            x = F.elu(x)
            x = self.pooling1(x)
            x = self.dropout1(x)
            
            x = self.separable_depth(x)
            x = self.separable_point(x)
            x = self.batchnorm3(x)
            x = F.elu(x)
            x = self.pooling2(x)
            x = self.dropout2(x)
            self.final_dim = x.numel()
            
        # Clasificador final
        self.fc = nn.Linear(self.final_dim, num_classes)

    def forward(self, x):
        # x entra como (Batch, 18, 256). EEGNet necesita (Batch, 1, 18, 256)
        x = x.unsqueeze(1)
        
        # Bloque 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        
        # Bloque 2
        x = self.depthwise1(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.pooling1(x)
        x = self.dropout1(x)
        
        # Bloque 3
        x = self.separable_depth(x)
        x = self.separable_point(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.pooling2(x)
        x = self.dropout2(x)
        
        # Flatten y Clasificación
        x = x.view(x.size(0), -1)
        return self.fc(x)