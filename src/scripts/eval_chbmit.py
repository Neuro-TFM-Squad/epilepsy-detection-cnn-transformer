import sys
import os

# Forzamos a Python a incluir la carpeta 'src' en su radar de búsqueda
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from models.baseline_model import BaselineCNN
from scripts.seizuredatasetnoise import SeizureDataset
from tqdm import tqdm

def main():
    # -------------------------
    # Configuración
    # -------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "models/baseline_cnn_bonn_noise.pth"

    # -------------------------
    # Dataset y DataLoader
    # -------------------------
    full_test_dataset = SeizureDataset(
        csv_file="data/CHB-MIT/processed/chbmit_index.csv",
        root_dir="data/CHB-MIT/processed/",
        augment=False # Importante: sin ruido en test
    )

    # Extraemos el 10% manteniendo la proporción de crisis/no-crisis
    labels = full_test_dataset.annotations['label'].values
    indices = list(range(len(full_test_dataset)))

    _, subset_idx = train_test_split(
        indices, 
        test_size=0.1,       # Cogemos solo el 10%
        random_state=42, 
        stratify=labels
    )

    test_dataset = Subset(full_test_dataset, subset_idx)

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,  # Aumentamos workers para acelerar la carga
        pin_memory=True # Mejora la transferencia a GPU
    )

    # -------------------------
    # Modelo
    # -------------------------
    model = BaselineCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    criterion = nn.BCEWithLogitsLoss()

    # -------------------------
    # Evaluación
    # -------------------------
    model.eval()
    total_loss = 0.0
    num_batches = 0

    all_logits = []
    all_labels = []

    with torch.no_grad():
        
        MAX_SAMPLES = int(0.1 * len(test_loader.dataset))
        seen_samples = 0

        
        for x, y in tqdm(test_loader, desc="Evaluando CHB-MIT (subsample)"):
            batch_size = x.size(0)
            if seen_samples >= MAX_SAMPLES:
                break

            x = x.to(device)
            y = y.to(device).float()

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item()
            num_batches += 1

            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())
            seen_samples += batch_size

    avg_loss = total_loss / num_batches

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)


    probs = torch.sigmoid(all_logits)
    preds = (probs >= 0.5).float()

    # Accuracy
    accuracy = (preds == all_labels).float().mean()

    # Sensibilidad (Recall clase crisis = 1)
    true_positives = ((preds == 1) & (all_labels == 1)).sum()
    false_negatives = ((preds == 0) & (all_labels == 1)).sum()

    sensitivity = true_positives / (true_positives + false_negatives + 1e-8)

    print(f"Accuracy:     {accuracy.item():.4f}")
    print(f"Sensitivity:  {sensitivity.item():.4f}")


    print(f"✅ Evaluación en CHBMIT completada")
    print(f"Loss media: {avg_loss:.4f}")
    print(f"Número de muestras evaluadas: {len(all_labels)}")

if __name__ == "__main__":
    main()