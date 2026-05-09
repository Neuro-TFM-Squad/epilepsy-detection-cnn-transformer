import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatioTemporalCNN(nn.Module):
    def __init__(self, in_channels=18, window_size=256, temporal_filters=16, spatial_filters=32, num_classes=1):
        # 1. Inicializamos la clase padre PRIMERO
        super(SpatioTemporalCNN, self).__init__()
        
        self.in_channels = in_channels
        self.window_size = window_size
        
        # ==========================================
        # 2. BLOQUE TEMPORAL (Conv1d con groups)
        # ==========================================
        self.temporal_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels * temporal_filters,
            kernel_size=15,
            padding=(15-1)//2, # Padding dinámico exacto
            groups=in_channels,
            bias=False
        )
        self.temporal_shortcut = nn.Conv1d(
            in_channels, 
            in_channels * temporal_filters, 
            kernel_size=1, 
            groups=in_channels
        )
        self.bn_temp = nn.BatchNorm1d(in_channels * temporal_filters)
        
        # ==========================================
        # 3. BLOQUE ESPACIAL (Conv1d cruzando canales)
        # ==========================================
        self.spatial_conv = nn.Conv1d(
            in_channels=in_channels * temporal_filters,
            out_channels=spatial_filters,
            kernel_size=1,
            groups=1,
            bias=False
        )
        self.spatial_shortcut = nn.Conv1d(
            in_channels * temporal_filters, 
            spatial_filters, 
            kernel_size=1
        )
<<<<<<< HEAD
        self.bn_spat = nn.BatchNorm1d(spatial_filters)
=======
        self.bn_spat = nn.InstanceNorm1d(spatial_filters, affine=True)
>>>>>>> master
        
        # ==========================================
        # 4. REDUCCIÓN Y EXTRACCIÓN (Pooling)
        # ==========================================
        self.pool = nn.AvgPool1d(kernel_size=8, stride=8)
<<<<<<< HEAD
        self.dropout = nn.Dropout(p=0.5)
=======
        self.spatial_dropout = nn.Dropout1d(p=0.1)
>>>>>>> master
        
        self.conv_refine = nn.Conv1d(
            spatial_filters, 
            spatial_filters * 2, 
            kernel_size=3, 
            padding=1
        )
<<<<<<< HEAD
        self.bn_refine = nn.BatchNorm1d(spatial_filters * 2)
=======
        self.bn_refine = nn.InstanceNorm1d(spatial_filters * 2, affine=True)
>>>>>>> master
        self.refine_shortcut = nn.Conv1d(
            spatial_filters, 
            spatial_filters * 2, 
            kernel_size=1
        )
        self.pool2 = nn.AvgPool1d(kernel_size=4, stride=4)
<<<<<<< HEAD
=======
        self.dropout = nn.Dropout(p=0.3)
>>>>>>> master
        
        # ==========================================
        # 5. CÁLCULO DINÁMICO DE LA DIMENSIÓN FINAL
        # ==========================================
        # Pasamos el tensor dummy por las capas ya creadas para obtener el tamaño aplanado exacto
        dummy = torch.zeros(1, in_channels, window_size)
        with torch.no_grad():
            res1 = self.temporal_shortcut(dummy)
            x = self.temporal_conv(dummy)
            x = F.elu(self.bn_temp(x))
            x = F.elu(x + res1)
            
            res2 = self.spatial_shortcut(x)
            x = self.spatial_conv(x)
            x = F.elu(self.bn_spat(x))
            x = F.elu(x + res2)
            
            x = self.pool(x)
            
            res3 = self.refine_shortcut(x)
            x = self.conv_refine(x)
            x = F.elu(self.bn_refine(x))
            x = F.elu(x + res3)
            
            x = self.pool2(x)
            
            # Extraemos la dimensión resultante
            self.final_feature_dim = x.numel() 
        
        # ==========================================
        # 6. CAPA DE CLASIFICACIÓN FINAL
        # ==========================================
        self.fc = nn.Linear(self.final_feature_dim, num_classes)

    def forward(self, x, return_embedding=False):
        # INPUT: (B, 18, 256)
        
        # ===== BLOQUE 1: TEMPORAL =====
        residual1 = self.temporal_shortcut(x)
        x = self.temporal_conv(x)
        x = F.elu(self.bn_temp(x))
        x = x + residual1
        x = F.elu(x)
        
        # ===== BLOQUE 2: ESPACIAL =====  
        residual2 = self.spatial_shortcut(x)
        x = self.spatial_conv(x)
        x = F.elu(self.bn_spat(x))
        x = x + residual2
        x = F.elu(x)
        
        # ===== POOL 1 =====
        x = self.pool(x)
<<<<<<< HEAD
        x = self.dropout(x)
=======
        x = self.spatial_dropout(x)
>>>>>>> master
        
        # ===== BLOQUE 3: REFINE =====
        residual3 = self.refine_shortcut(x)
        x = self.conv_refine(x)
        x = F.elu(self.bn_refine(x))
        x = x + residual3  
        x = F.elu(x)
        
        # ===== POOL 2 + FLATTEN =====
        x = self.pool2(x)
<<<<<<< HEAD
        x = self.dropout(x)
=======
        x = self.spatial_dropout(x)
>>>>>>> master
        x_flat = x.view(x.size(0), -1)
        
        # ===== SALIDA =====
        # Si la usamos dentro de LSTM/Transformer, cortamos antes de la FC
        if return_embedding:
            return x_flat
<<<<<<< HEAD
            
=======
        x_flat = self.dropout(x_flat)    
>>>>>>> master
        return self.fc(x_flat)
        
    def get_spatial_attention(self, x_after_spatial):
        # ATENCIÓN: Esto calcula la relevancia sobre los filtros abstractos espaciales (ej. 32),
        # no sobre los 18 canales topológicos originales, ya que esos se mezclaron en el bloque espacial.
        attn = torch.softmax(x_after_spatial.mean(dim=-1), dim=1)
        return attn