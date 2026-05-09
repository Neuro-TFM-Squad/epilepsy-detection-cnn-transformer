import torch
import torch.nn as nn
import torch.optim as optim
from models.baseline_model import BaselineCNN
from scripts.seizuredataset import SeizureDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BaselineCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
train_dataset = SeizureDataset(
    csv_file="data/BONN/processed/bonn_index.csv",
    root_dir="data/BONN/processed/"
)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)


model.train()

for epoch in range(5):  # UNA época
    running_loss = 0.0

    for x, y in train_loader:
        x = x.to(device)                  # [B, C, 256]
        y = y.to(device).float()           # [B]

        optimizer.zero_grad()
        logits = model(x)                  # [B]
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")
    
import os
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/baseline_cnn_bonn.pth")
