import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Asegurar rutas
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.baseline_model import BaselineCNN
from scripts.seizuredataset_h5 import SeizureDatasetH5

def extract_probabilities(model, dataloader, device):
    """Pasa la red neuronal una sola vez y extrae la matriz de (N_ventanas, 18_canales)"""
    all_probs = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Extrayendo probabilidades de la CNN"):
            B, C, L = x.shape
            # Aplanamos para la CNN: (B*18, 1, 256)
            x_flat = x.view(B * C, 1, L).to(device)
            logits = model(x_flat).squeeze()
            
            # Reconstruimos la forma: (B, 18) -> Cada fila es 1 segundo, cada columna 1 canal
            probs = torch.sigmoid(logits).view(B, C)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.numpy()) # y es (B,)
            
    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)

def find_events(labels):
    """Encuentra los índices de inicio y fin de las crisis reales"""
    padded = np.pad(labels, (1, 1), 'constant')
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    return list(zip(starts, ends))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Iniciando Grid Search Heurístico en {device}...")

    # Rutas
    model_weights = "models/baseline_cnn_chbmit_finetuned.pth"
    h5_path = "data/CHB-MIT/processed/chbmit_dataset.h5"
    
    # 1. Cargar Datos (Manteniendo orden cronológico en validación)
    val_dataset = SeizureDatasetH5(h5_path=h5_path, augment=False)
    labels = val_dataset.labels
    indices = list(range(len(val_dataset)))
    
    _, val_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)
    # ORDEN CRONOLÓGICO VITAL PARA EL FILTRO TEMPORAL
    val_idx = sorted(val_idx) 
    
    val_loader = DataLoader(Subset(val_dataset, val_idx), batch_size=128, shuffle=False, num_workers=4)

    # 2. Cargar Modelo
    model = BaselineCNN().to(device)
    model.load_state_dict(torch.load(model_weights, map_location=device, weights_only=True))

    # 3. Extracción (Solo se hace una vez)
    probs_matrix, true_labels = extract_probabilities(model, val_loader, device)
    
    # Eventos reales
    true_events = find_events(true_labels)
    total_events = len(true_events)
    print(f"\n📊 Total de eventos de crisis reales a detectar: {total_events}")

    # 4. Grid Search
    base_thresh = 0.7088 # El umbral que sacamos en el script anterior para mantener 50% sens.
    K_range = range(1, 11) # De 1 a 10 canales
    T_range = range(1, 6)  # De 1 a 5 segundos
    
    results = []
    
    # Pre-calculamos la máscara base (muy rápido)
    binary_matrix = (probs_matrix >= base_thresh).astype(int)
    # Sumamos los canales que pitan en cada segundo
    votes_per_second = binary_matrix.sum(axis=1)

    print("🔎 Ejecutando combinaciones K y T...")
    for K in K_range:
        # Filtro Espacial
        candidates = (votes_per_second >= K).astype(int)
        
        for T in T_range:
            # Filtro Temporal (Ventana deslizante)
            # Convolvemos con un array de unos de tamaño T
            temporal_sum = np.convolve(candidates, np.ones(T), mode='valid')
            
            # Rellenamos el principio con ceros para mantener el tamaño original
            alarms = np.zeros_like(candidates)
            alarms[T-1:] = (temporal_sum >= T).astype(int)
            
            # --- Métricas por Ventana ---
            tp = ((alarms == 1) & (true_labels == 1)).sum()
            fp = ((alarms == 1) & (true_labels == 0)).sum()
            fn = ((alarms == 0) & (true_labels == 1)).sum()
            
            prec = tp / (tp + fp + 1e-8)
            rec = tp / (tp + fn + 1e-8)
            f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
            
            # --- Métricas por Evento ---
            detected_events = 0
            for start, end in true_events:
                # Si la alarma suena en CUALQUIER momento de la crisis real, es un acierto
                if alarms[start:end+1].any():
                    detected_events += 1
                    
            event_sens = detected_events / total_events if total_events > 0 else 0
            
            results.append({
                'K (Canales)': K,
                'T (Segundos)': T,
                'Precisión (%)': prec * 100,
                'Sens. Ventana (%)': rec * 100,
                'F1-Score': f1,
                'Sens. EVENTO (%)': event_sens * 100
            })

    # 5. Mostrar Resultados
    df = pd.DataFrame(results)
    # Ordenamos por F1-Score para ver las mejores configuraciones arriba
    df_sorted = df.sort_values(by='F1-Score', ascending=False).head(10)
    
    print("\n🏆 TOP 10 MEJORES CONFIGURACIONES (Ordenadas por F1-Score):")
    print("-" * 80)
    print(df_sorted.to_string(index=False, float_format="%.2f"))
    print("-" * 80)

if __name__ == '__main__':
    main()