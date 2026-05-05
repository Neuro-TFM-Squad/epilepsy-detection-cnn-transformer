import sys
import os

# Forzamos a Python a incluir la carpeta 'src' en su radar de búsqueda
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.baseline_model import BaselineCNN
from scripts.seizuredatasetnoise import SeizureDataset

# -------------------------
# Configuración
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models/baseline_cnn_bonn_noise.pth"

# -------------------------
# Dataset y DataLoader
# -------------------------
test_dataset = SeizureDataset(
    csv_file="data/BONN/processed/bonn_index.csv",
    root_dir="data/BONN/processed/",
    augment=False # Importante: sin ruido en test
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

# -------------------------
# Modelo
# -------------------------
model = BaselineCNN().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))

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
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device).float()

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item()
        num_batches += 1

        all_logits.append(logits.cpu())
        all_labels.append(y.cpu())

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


print(f"✅ Evaluación en Bonn completada")
print(f"Loss media: {avg_loss:.4f}")
print(f"Número de muestras evaluadas: {len(all_labels)}")