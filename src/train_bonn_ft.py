import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Asegurar que las rutas de tus modelos y scripts sean accesibles
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.baseline_model import BaselineCNN
from scripts.seizuredataset_h5 import SeizureDatasetH5 

def calculate_metrics(tp, fp, fn):
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    return precision, recall, f1

def main():
    # ==========================================
    # 1. CONFIGURACIÓN
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bonn_weights = "models/baseline_cnn_bonn_noise.pth"
    h5_path = "data/CHB-MIT/processed/chbmit_dataset.h5"
    save_path = "models/baseline_cnn_chbmit_finetuned.pth"
    
    # Hiperparámetros
    batch_size = 128
    max_epochs = 50
    patience = 7
    lr_backbone = 1e-5
    lr_classifier = 1e-4

    print(f"Iniciando Fine-Tuning en {device}...")

    # ==========================================
    # 2. DATOS Y BALANCEO (WeightedRandomSampler)
    # ==========================================
    # Usamos aumento de datos en train para mejorar robustez morfológica
    full_dataset = SeizureDatasetH5(h5_path=h5_path, augment=True)
    val_dataset_clean = SeizureDatasetH5(h5_path=h5_path, augment=False)

    labels = full_dataset.labels
    indices = list(range(len(full_dataset)))

    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)
    
    # Sampler para balanceo 50/50 en entrenamiento
    train_labels = labels[train_idx]
    class_counts = np.bincount(train_labels)
    weights = 1. / class_counts
    samples_weights = torch.from_numpy(np.array([weights[l] for l in train_labels]))
    
    muestras_por_epoca = 32000 # Esto te dará 1000 iteraciones por época con batch_size=32
    sampler = WeightedRandomSampler(
        weights=samples_weights, 
        num_samples=muestras_por_epoca, 
        replacement=True
    )

    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(Subset(val_dataset_clean, val_idx), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # ==========================================
    # 3. MODELO Y OPTIMIZADOR DIFERENCIAL
    # ==========================================
    model = BaselineCNN().to(device)
    model.load_state_dict(torch.load(bonn_weights, map_location=device, weights_only=True))

    # Identificar parámetros (Asumiendo que la última capa se llama 'linear')
    classifier_params = []
    backbone_params = []
    
    for name, param in model.named_parameters():
        if 'linear' in name: # Ajustar si tu capa de salida tiene otro nombre (ej. 'fc')
            classifier_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': lr_backbone},
        {'params': classifier_params, 'lr': lr_classifier}
    ])

    criterion = nn.BCEWithLogitsLoss() # pos_weight = 1.0 por defecto

    # ==========================================
    # 4. LOOP DE ENTRENAMIENTO
    # ==========================================
    best_f1 = 0.0
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        # --- Fase de Entrenamiento ---
        model.train()
        train_loss, t_tp, t_fp, t_fn = 0.0, 0, 0, 0
        
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1} Train", leave=False):
            # Adaptar forma: (B, 18, 256) -> (B*18, 1, 256)
            B, C, L = x.shape
            x = x.view(B * C, 1, L).to(device)
            y = y.repeat_interleave(C).to(device).float()

            optimizer.zero_grad()
            logits = model(x).squeeze()
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (torch.sigmoid(logits) >= 0.5).float()
            t_tp += ((preds == 1) & (y == 1)).sum().item()
            t_fp += ((preds == 1) & (y == 0)).sum().item()
            t_fn += ((preds == 0) & (y == 1)).sum().item()

        t_prec, t_rec, t_f1 = calculate_metrics(t_tp, t_fp, t_fn)

        # --- Fase de Validación ---
        # --- Fase de Validación ---
        model.eval()
        val_loss, v_tp, v_fp, v_fn = 0.0, 0, 0, 0
        
        max_val_batches = 300 
        batches_procesados = 0
        
        with torch.no_grad():
            # El bloque 'with' soluciona el cuelgue visual al hacer 'break'
            with tqdm(val_loader, desc=f"Epoch {epoch+1} Val", leave=False, total=min(max_val_batches, len(val_loader))) as val_bar:
                for i, (x, y) in enumerate(val_bar):
                    if i >= max_val_batches:
                        break 
                    
                    B, C, L = x.shape
                    x = x.view(B * C, 1, L).to(device)
                    y = y.repeat_interleave(C).to(device).float()

                    logits = model(x).squeeze()
                    loss = criterion(logits, y)
                    val_loss += loss.item()

                    preds = (torch.sigmoid(logits) >= 0.5).float()
                    v_tp += ((preds == 1) & (y == 1)).sum().item()
                    v_fp += ((preds == 1) & (y == 0)).sum().item()
                    v_fn += ((preds == 0) & (y == 1)).sum().item()
                    
                    batches_procesados += 1
        
        # Calculamos la loss y métricas con los batches reales que hemos procesado
        avg_val_loss = val_loss / batches_procesados
        v_prec, v_rec, v_f1 = calculate_metrics(v_tp, v_fp, v_fn)

        # Monitorización por consola
        print(f"Epoch {epoch+1:02d} | Loss: {avg_val_loss:.4f} | "
                f"Prec: {v_prec:.4f} | Rec (Sens): {v_rec:.4f} | F1: {v_f1:.4f}")

        # --- Early Stopping basado en F1-Score ---
        if v_f1 > best_f1:
            best_f1 = v_f1
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            print(f"  ⭐ Nuevo mejor F1: {best_f1:.4f}. Modelo guardado.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"🛑 Early stopping tras {patience} épocas sin mejora en F1.")
                break

    print(f"\n✅ Fine-tuning completado. Mejor F1 obtenido: {best_f1:.4f}")

if __name__ == '__main__':
    main()