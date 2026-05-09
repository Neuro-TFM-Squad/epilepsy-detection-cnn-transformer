import os
import sys

# Esto añade la carpeta raíz del proyecto al PATH de Python temporalmente
# Asume que este script está dentro de la carpeta 'src'
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm

# Tus importaciones locales
from src.datasets.seizureDatasetMulticanal import SeizureDatasetMultichannel
from src.models.baseline_model_multicanal import SpatioTemporalCNN
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    """
    Focal Loss para problemas binarios con desbalanceo extremo.
    alpha: Pondera la importancia de la clase positiva (crisis). 
           Si es > 0.5, le da más peso a no fallar las crisis.
    gamma: El factor de enfoque. Cuanto más alto (ej. 2 o 3), más castiga 
           a la red por equivocarse en ejemplos "fáciles" y más se centra en los "difíciles" (falsos positivos).
    """
    def __init__(self, alpha=0.75, gamma=2.0):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        # 1. Calculamos la probabilidad aplicando sigmoide
        probs = torch.sigmoid(logits)
        
        # 2. Calculamos la pérdida binaria clásica (BCE) por cada ventana individualmente
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # 3. p_t es la probabilidad de que el modelo haya acertado
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # 4. Aplicamos el peso alpha (favorece a la clase 1)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # 5. FÓRMULA FOCAL LOSS: alpha_t * (1 - p_t)^gamma * BCE
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        focal_loss = focal_weight * bce_loss
        
        return focal_loss.mean()

def inject_bonn_knowledge(chb_model, bonn_model_path, device, num_channels=18):
    bonn_state_dict = torch.load(bonn_model_path, map_location=device, weights_only=True)
    
    # 1. Extraemos pesos de la Convolución y los clonamos 18 veces
    bonn_conv_weights = bonn_state_dict['block1.0.weight'] 
    repeated_conv = bonn_conv_weights.repeat(num_channels, 1, 1) 
    
    with torch.no_grad():
        # Inyectamos en la capa convolucional
        chb_model.temporal_conv.weight.copy_(repeated_conv)
        
        # 2. Transferencia del BatchNorm
        bn_weight = bonn_state_dict['block1.1.weight'].repeat(num_channels)
        bn_bias = bonn_state_dict['block1.1.bias'].repeat(num_channels)
        bn_mean = bonn_state_dict['block1.1.running_mean'].repeat(num_channels)
        bn_var = bonn_state_dict['block1.1.running_var'].repeat(num_channels)
        
        chb_model.bn_temp.weight.copy_(bn_weight)
        chb_model.bn_temp.bias.copy_(bn_bias)
        chb_model.bn_temp.running_mean.copy_(bn_mean)
        chb_model.bn_temp.running_var.copy_(bn_var)
        
    # 3. Congelamos la Convolución y el BatchNorm temporal
    chb_model.temporal_conv.weight.requires_grad = False
    for param in chb_model.bn_temp.parameters():
        param.requires_grad = False
        
    chb_model.bn_temp.eval() 
    
    return chb_model

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    # Forzamos que el BatchNorm temporal siga en modo evaluación para no estropear lo aprendido en Bonn
    model.bn_temp.eval() 
    
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Entrenando", leave=False)
    for signals, labels in pbar:
        signals = signals.to(device)
        # BCEWithLogitsLoss espera etiquetas en formato float y con forma [Batch, 1]
        labels = labels.unsqueeze(1).float().to(device)
        
        optimizer.zero_grad()
        
        logits = model(signals)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Para las métricas: aplicamos sigmoide y umbral 0.5
        probs = torch.sigmoid(logits).detach().cpu()
        preds = (probs >= 0.5).int().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix(loss=loss.item())
        
    epoch_loss = running_loss / len(dataloader)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return epoch_loss, f1

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validando", leave=False)
        for signals, labels in pbar:
            signals = signals.to(device)
            labels = labels.unsqueeze(1).float().to(device)
            
            logits = model(signals)
            loss = criterion(logits, labels)
            running_loss += loss.item()
            
            probs = torch.sigmoid(logits).cpu()
            preds = (probs >= 0.5).int().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
    epoch_loss = running_loss / len(dataloader)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return epoch_loss, prec, rec, f1

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # Optimización para entradas de tamaño fijo
    print(f"🚀 Iniciando entrenamiento en {device}")

    # 1. Rutas
    csv_path = os.path.join(ROOT_DIR, "data", "CHBMIT", "processed", "chbmit_index.csv")
    index_df = pd.read_csv(csv_path)
    
    # Nuevas rutas a los archivos binarios del subset
    signals_bin = os.path.join(ROOT_DIR, "data", "CHBMIT", "processed", "subset_signals.bin")
    labels_bin = os.path.join(ROOT_DIR, "data", "CHBMIT", "processed", "subset_labels.bin")

    ruta_pesos_bonn = os.path.join(ROOT_DIR, "models", "baseline_cnn_bonn_noise_k15.pth")
    save_path = os.path.join(ROOT_DIR, "models", "spatio_temporal_cnn.pth")
    # ... (rutas de modelos, igual que antes) ...

    # ==========================================================
    # 2. Configuración de Pacientes y Split (VERSIÓN SUBSET)
    # ==========================================================
    index_df['patient_id'] = index_df['filepath'].apply(lambda x: x.split('_')[0])
    
    target_patients = ['chb01', 'chb02', 'chb03', 'chb04', 'chb05', 'chb06']
    
    # 2.1 Recreamos el orden EXACTO en el que el script generó el .bin
    dfs_filtrados = []
    for p in target_patients:
        dfs_filtrados.append(index_df[index_df['patient_id'] == p])
        
    # 2.2 Reseteamos el índice. Ahora chb01 empieza en 0, hasta llegar al último paciente.
    subset_df = pd.concat(dfs_filtrados).reset_index(drop=True)
    
    # 2.3 Ahora sí, separamos Train y Val usando los índices nuevos
    train_patients = ['chb01', 'chb02', 'chb03', 'chb04'] 
    val_patients = ['chb05', 'chb06']  
    
    train_idx = subset_df[subset_df['patient_id'].isin(train_patients)].index.tolist()
    val_idx = subset_df[subset_df['patient_id'].isin(val_patients)].index.tolist()

    print(f"🏥 Pacientes en Train: {len(train_patients)} (Ventanas: {len(train_idx)})")
    print(f"🏥 Pacientes en Validation: {len(val_patients)} (Ventanas: {len(val_idx)})")

    # 3. Cálculo del pos_weight (Usando el subset_df)
    train_labels = subset_df.loc[train_idx, 'label'].values
    num_positives = np.sum(train_labels == 1)
    num_negatives = np.sum(train_labels == 0)
    peso_crisis = num_negatives / num_positives if num_positives > 0 else 1.0
    print(f"⚖️ Peso dinámico para la Loss (pos_weight): {peso_crisis:.2f}")

    # 4. Datasets y DataLoaders
    train_dataset = SeizureDatasetMultichannel(signals_path=signals_bin, labels_path=labels_bin, augment=True, sequence_length=1)
    val_dataset = SeizureDatasetMultichannel(signals_path=signals_bin, labels_path=labels_bin, augment=False, sequence_length=1)
    
    train_loader = DataLoader(Subset(train_dataset, train_idx), batch_size=512, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(Subset(val_dataset, val_idx), batch_size=512, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

    # 5. Modelo y Transfer Learning
    model = SpatioTemporalCNN(in_channels=18).to(device)
    model = inject_bonn_knowledge(model, ruta_pesos_bonn, device)
    print("✅ Pesos inyectados y bloque temporal congelado.")

    # 6. Optimizador y Loss
    criterion = BinaryFocalLoss(alpha=0.85, gamma=2.0).to(device)
    parametros_entrenables = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parametros_entrenables, lr=1e-4, weight_decay=1e-5)

    # ==========================================
    # 7. BUCLE DE ENTRENAMIENTO PRINCIPAL
    # ==========================================
    epochs = 150
    best_val_f1 = 0.0
    patience = 10
    patience_counter = 0
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("\n🔥 Comenzando épocas de entrenamiento...")
    for epoch in range(epochs):
        print(f"\n--- Época {epoch+1}/{epochs} ---")
        
        train_loss, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_prec, val_rec, val_f1 = validate(model, val_loader, criterion, device)
        
        print(f"Train | Loss: {train_loss:.4f} | F1: {train_f1:.4f}")
        print(f"Val   | Loss: {val_loss:.4f} | F1: {val_f1:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f}")
        
        # --- EARLY STOPPING LOGIC ---
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0 # Reseteamos la paciencia
            torch.save(model.state_dict(), save_path)
            print(f"⭐ ¡Nuevo mejor modelo guardado! (F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
            print(f"⏳ Sin mejora en F1-Score (Paciencia: {patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print(f"\n[!] Early Stopping activado en la época {epoch+1}. El modelo ha dejado de mejorar.")
                break # Detiene el entrenamiento

    print("\n🎉 Entrenamiento finalizado. Mejor F1 de validación:", best_val_f1)

if __name__ == '__main__':
    main()