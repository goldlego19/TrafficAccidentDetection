import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import logging
from tqdm import tqdm # --- IMPORTED TQDM ---

# --- CONFIG ---
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'optical_flow_maps')
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 32
EPOCHS = 30 
# --------------

logging.basicConfig(level=logging.INFO, format='%(message)s')

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"🚀 STARTING FROZEN RESNET18 ON {device}")

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_train_data = datasets.ImageFolder(root=DATA_DIR, transform=train_transform)
    full_val_data = datasets.ImageFolder(root=DATA_DIR, transform=val_transform)
    
    accident_idx = full_train_data.class_to_idx.get('accident', 0)
    
    dataset_size = len(full_train_data)
    indices = torch.randperm(dataset_size).tolist()
    train_size = int(0.8 * dataset_size)

    train_dataset = Subset(full_train_data, indices[:train_size])
    val_dataset = Subset(full_val_data, indices[train_size:])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    model = model.to(device)

    optimizer = optim.Adam(model.fc.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.BCELoss()
    best_f1 = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        # --- TRAINING PROGRESS BAR ---
        # leave=False makes it disappear cleanly after the epoch finishes
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [Train]", leave=False)
        
        for X, y in train_pbar:
            X, y = X.to(device), (y == accident_idx).float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            # Update the progress bar with the current live loss
            train_pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        model.eval()
        tp, fp, tn, fn = 0, 0, 0, 0
        
        # --- VALIDATION PROGRESS BAR ---
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [Val]  ", leave=False)
        
        with torch.no_grad():
            for X, y in val_pbar:
                X, y = X.to(device), (y == accident_idx).float().unsqueeze(1).to(device)
                preds = (model(X) > 0.5).float()
                tp += ((preds == 1) & (y == 1)).sum().item()
                fp += ((preds == 1) & (y == 0)).sum().item()
                tn += ((preds == 0) & (y == 0)).sum().item()
                fn += ((preds == 0) & (y == 1)).sum().item()

        p = tp / (tp + fp + 1e-8)
        r = tp / (tp + fn + 1e-8)
        f1 = 2 * (p * r) / (p + r + 1e-8)
        avg_loss = total_loss / len(train_loader)
        
        logging.info(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | F1: {f1:.3f} | P: {p:.2f} R: {r:.2f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'best_resnet_model.pth'))
            logging.info(f"  ⭐ New Best Model Saved! (F1: {f1:.3f})")

if __name__ == '__main__':
    train()