import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Asegurar rutas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.baseline_model import BaselineCNN
from scripts.seizuredataset_h5 import SeizureDatasetH5

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔍 Iniciando calibración de umbral en {device}...")

    # Rutas
    model_weights = "models/baseline_cnn_chbmit_finetuned.pth"
    h5_path = "data/CHB-MIT/processed/chbmit_dataset.h5"

    # 1. Cargar Dataset de Validación (Igual que en el entrenamiento)
    val_dataset_clean = SeizureDatasetH5(h5_path=h5_path, augment=False)
    labels = val_dataset_clean.labels
    indices = list(range(len(val_dataset_clean)))
    _, val_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)
    
    val_loader = DataLoader(
        Subset(val_dataset_clean, val_idx), 
        batch_size=128, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )

    # 2. Cargar Modelo
    model = BaselineCNN().to(device)
    model.load_state_dict(torch.load(model_weights, map_location=device, weights_only=True))
    model.eval()

    # 3. Extracción de Probabilidades y Etiquetas Reales
    all_probs = []
    all_labels = []

    # Limitamos la validación a unos 500 batches para no esperar horas, 
    # es muestra más que suficiente para trazar una curva estadística fiable.
    max_batches = 500 

    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(val_loader, desc="Extrayendo probabilidades")):
            if i >= max_batches: break
            
            B, C, L = x.shape
            x = x.view(B * C, 1, L).to(device)
            y = y.repeat_interleave(C).to(device).float()

            logits = model(x).squeeze()
            probs = torch.sigmoid(logits)  # Convertimos logits a probabilidades (0 a 1)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # 4. Cálculo de la Curva Precision-Recall
    print("📈 Calculando Curva PR...")
    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)

    # El F1-Score requiere cálculo manual desde precisions y recalls
    # Añadimos 1e-8 para evitar divisiones por cero
    f1_scores = 2 * (precisions[:-1] * recalls[:-1]) / (precisions[:-1] + recalls[:-1] + 1e-8)

    # A) Buscar el umbral que MAXIMIZA el F1-Score
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    best_prec = precisions[best_idx]
    best_rec = recalls[best_idx]

    # B) Buscar el umbral para mantener al menos un 50% de Sensibilidad (Recall >= 0.50)
    # Buscamos el índice donde el recall es lo más cercano a 0.50 sin bajar de ahí
    valid_recall_indices = np.where(recalls[:-1] >= 0.50)[0]
    if len(valid_recall_indices) > 0:
        # Cogemos el que tenga mayor precisión dentro de los que cumplen
        target_idx = valid_recall_indices[np.argmax(precisions[valid_recall_indices])]
        target_thresh = thresholds[target_idx]
        target_prec = precisions[target_idx]
        target_rec = recalls[target_idx]
    else:
        target_thresh, target_prec, target_rec = 0, 0, 0

    print("\n" + "="*40)
    print("🏆 RESULTADOS DE LA CALIBRACIÓN")
    print("="*40)
    print(f"Umbral Óptimo (Max F1):   {best_threshold:.4f}")
    print(f" -> Precisión: {best_prec:.4f} | Sensibilidad: {best_rec:.4f} | F1: {best_f1:.4f}\n")
    print(f"Umbral para Sensibilidad >= 50%: {target_thresh:.4f}")
    print(f" -> Precisión: {target_prec:.4f} | Sensibilidad: {target_rec:.4f}")
    print("="*40)

    # 5. Visualización y Guardado
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, color='blue', label='PR Curve')
    plt.scatter([best_rec], [best_prec], color='red', marker='o', s=100, zorder=5, label=f'Max F1 (Thr={best_threshold:.2f})')
    
    if len(valid_recall_indices) > 0:
        plt.scatter([target_rec], [target_prec], color='green', marker='X', s=100, zorder=5, label=f'50% Sens (Thr={target_thresh:.2f})')

    plt.title('Curva Precision-Recall (Validación CHB-MIT)')
    plt.xlabel('Sensibilidad (Recall)')
    plt.ylabel('Precisión')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plot_path = "models/pr_curve.png"
    plt.savefig(plot_path)
    print(f"\n📊 Gráfica guardada en: {plot_path}")

if __name__ == '__main__':
    main()