import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # <-- Importamos tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, DataLoader

from models.baseline_model import BaselineCNN
from scripts.seizuredatasetnoise import SeizureDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Iniciando entrenamiento en dispositivo: {device}")

# ==========================================
# 1. CONFIGURACIÓN DE DATOS
# ==========================================
csv_path = "data/BONN/processed/bonn_index.csv"
root_data = "data/BONN/processed/"

dataset_aug = SeizureDataset(csv_file=csv_path, root_dir=root_data, augment=True)
dataset_clean = SeizureDataset(csv_file=csv_path, root_dir=root_data, augment=False)

labels = dataset_aug.annotations['label'].values
indices = list(range(len(dataset_aug)))

train_idx, val_idx = train_test_split(
    indices, 
    test_size=0.2, 
    random_state=42, 
    stratify=labels
)

train_dataset = Subset(dataset_aug, train_idx)
val_dataset = Subset(dataset_clean, val_idx)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ==========================================
# 2. MODELO Y OPTIMIZADOR
# ==========================================
model = BaselineCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

max_epochs = 150
patience = 15
patience_counter = 0
best_val_loss = float('inf')

os.makedirs("models", exist_ok=True)
best_model_path = "models/baseline_cnn_bonn_noise.pth"

# ==========================================
# 3. BUCLE DE ENTRENAMIENTO
# ==========================================
for epoch in range(max_epochs):
    # --- FASE DE ENTRENAMIENTO ---
    model.train()
    running_train_loss = 0.0

    # Envolvemos el dataloader con tqdm
    train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{max_epochs}] Train", leave=False)
    
    for x, y in train_bar:
        x = x.to(device)
        # Aseguramos que la etiqueta tenga la forma [B, 1] para BCEWithLogitsLoss
        y = y.to(device).float()
        
        optimizer.zero_grad()
        logits = model(x)
        
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()
        # Actualizamos la barra de progreso con el loss actual
        train_bar.set_postfix(loss=loss.item())

    avg_train_loss = running_train_loss / len(train_loader)

    # --- FASE DE VALIDACIÓN ---
    model.eval()
    running_val_loss = 0.0
    
    val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{max_epochs}] Val", leave=False)
    
    with torch.no_grad():
        for x, y in val_bar:
            x = x.to(device)
            y = y.to(device).float()
            
            logits = model(x)
            loss = criterion(logits, y)
            running_val_loss += loss.item()

    avg_val_loss = running_val_loss / len(val_loader)
    
    # Imprimimos el resumen de la época
    print(f"Epoch [{epoch+1}/{max_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # --- EARLY STOPPING LOGIC ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
        print("  -> ¡Mejor modelo guardado! (Val Loss reducida)")
    else:
        patience_counter += 1
        print(f"  -> Sin mejora (Paciencia: {patience_counter}/{patience})")
        if patience_counter >= patience:
            print(f"\n[!] Early stopping activado en la época {epoch+1}. El modelo ha dejado de aprender.")
            break
            
print("\n=== Entrenamiento Finalizado ===")
print(f"El mejor modelo se encuentra en: {best_model_path}")