import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score
import os
import glob

# --- 1. ARQUITECTURA: BASELINE 1D CNN (~50k params) ---
class SeizureNetBaseline(nn.Module):
    def __init__(self, num_classes=2):
        super(SeizureNetBaseline, self).__init__()
        
        # Entrada: [Batch, 1, 1024]
        self.features = nn.Sequential(
            # Bloque 1: Captura de transitorios rápidos
            nn.Conv1d(1, 16, kernel_size=7, stride=1, padding=3), 
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2), # -> 512
            
            # Bloque 2: Patrones rítmicos (Theta/Delta)
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2), # -> 256
            
            # Bloque 3: Características complejas
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2), # -> 128
            
            # Bloque 4: Refinamiento
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # GAP: La clave de la flexibilidad. 
            # Convierte [64, 128] en [64, 1] independientemente de la longitud.
            nn.AdaptiveAvgPool1d(1) 
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# --- 2. DATALOADER Y PREPROCESADO ---
class BonnDataset(Dataset):
    def __init__(self, data, labels):
        # data: (N, 700), labels: (N,)
        self.data = torch.FloatTensor(data).unsqueeze(1) # Add channel dim
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        # Preprocesado: Baseline Removal & Z-Score
        x = x - torch.mean(x)
        x = x / (torch.std(x) + 1e-6)
        return x, self.labels[idx]

# --- 3. UTILS: EARLY STOPPING ---
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose: print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss



def load_processed_data(processed_path):
    X = np.load(os.path.join(processed_path, 'X_bonn_256Hz.npy'))
    y = np.load(os.path.join(processed_path, 'y_bonn_256Hz.npy'))
    return X, y

# --- 4. PIPELINE DE ENTRENAMIENTO ---
def train_baseline():
    # 1. RUTA ABSOLUTA (Basada en tu estructura)
    path_data = r"D:\TFM\epilepsy-detection-cnn-transformer\data\BONN\raw"
    
    print("Cargando datos de Bonn...")
    X, y = load_processed_data(r"D:\TFM\epilepsy-detection-cnn-transformer\data\BONN\processed")
    print(f"Dataset cargado: {X.shape[0]} ventanas de {X.shape[1]} muestras.")

    # 2. SPLIT (80% Train, 20% Val) - Stratify asegura equilibrio de clases
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # 3. CREAR DATALOADERS
    train_ds = BonnDataset(X_train, y_train)
    val_ds = BonnDataset(X_val, y_val)
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SeizureNetBaseline(num_classes=2).to(device)
    
    # Siguiendo especificaciones: BCEWithLogitsLoss para Bonn
    # (Usamos CrossEntropyLoss aquí por facilidad con num_classes=2, es equivalente)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    early_stopping = EarlyStopping(patience=10, path='baseline_bonn.pt')

    for epoch in range(50):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validación
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        f1 = f1_score(all_labels, all_preds)
        
        print(f"Epoch {epoch+1}/50 | Loss: {train_loss/len(train_loader):.4f} | Val Loss: {avg_val_loss:.4f} | F1: {f1:.4f}")
        
        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    return model

if __name__ == "__main__":
    train_baseline()