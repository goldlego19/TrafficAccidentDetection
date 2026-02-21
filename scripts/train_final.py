import sys
import os

# --- PATH FIX: Add the project root to Python's search path ---
# This tells Python to look one level up (..) for the 'src' folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# -------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import logging
from datetime import datetime
from src.accident_detection_model import AccidentDetectionLSTM

# --- CONFIG ---
# Note: Since we are running from 'scripts/', we point to '../feature_cache'
# OR we assume you run this command FROM the root folder.
# To be safe, let's use absolute paths relative to this script.
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # scripts/
ROOT_DIR = os.path.dirname(BASE_DIR)                  # TrafficAccidentDetection/

CACHE_FILE = os.path.join(ROOT_DIR, 'feature_cache', 'final_features.pkl')
LOG_DIR = os.path.join(ROOT_DIR, 'logs')
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoints')

BATCH_SIZE = 16
EPOCHS = 100
LR = 0.0005
# --------------

# Setup Directories
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Setup Logging
def setup_logger():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(LOG_DIR, f'training_log_{timestamp}.txt')
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(file_handler)
    

    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass 
            
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    
    return logger, log_filename

logger, log_file = setup_logger()

class CachedDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.keys = list(data.keys())
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, idx):
        item = self.data[self.keys[idx]]
        return (torch.FloatTensor(item['features']), 
                torch.FloatTensor([item['label']]))

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info("="*50)
    logger.info(f"🚀 STARTING TRAINING ON {device}")
    logger.info(f"📝 Log file: {log_file}")
    logger.info(f"⚙️  Config: Batch={BATCH_SIZE}, LR={LR}, Epochs={EPOCHS}")
    logger.info("="*50)
    
    if not os.path.exists(CACHE_FILE):
        logger.error(f"❌ Error: Cache file not found at: {CACHE_FILE}")
        logger.error("   Run 'python scripts/step2_cache_features.py' first!")
        return

    logger.info("📂 Loading Data...")
    with open(CACHE_FILE, 'rb') as f:
        data = pickle.load(f)
        
    # Split Data 80/20
    keys = list(data.keys())
    np.random.shuffle(keys)
    
    split_idx = int(len(keys) * 0.8)
    train_keys = keys[:split_idx]
    val_keys = keys[split_idx:]
    
    train_data = {k: data[k] for k in train_keys}
    val_data = {k: data[k] for k in val_keys}
    
    # Count stats
    train_labels = [d['label'] for d in train_data.values()]
    val_labels = [d['label'] for d in val_data.values()]
    
    logger.info(f"✅ Data Loaded:")
    logger.info(f"   Train: {len(train_data)} samples (Accidents: {sum(train_labels)}, Normal: {len(train_labels)-sum(train_labels)})")
    logger.info(f"   Val:   {len(val_data)} samples (Accidents: {sum(val_labels)}, Normal: {len(val_labels)-sum(val_labels)})")
    
    train_loader = DataLoader(CachedDataset(train_data), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(CachedDataset(val_data), batch_size=BATCH_SIZE)
    
    # Initialize Model
    model = AccidentDetectionLSTM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss() 
    
    best_f1 = 0.0
    
    logger.info("\n🏁 Begin Epochs...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Validation Phase
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
                
        # Metrics Calculation
        preds_bin = (np.array(all_preds) > 0.5).astype(int)
        targets = np.array(all_targets)
        
        acc = (preds_bin == targets).mean()
        
        # Confusion Matrix elements
        tp = ((preds_bin == 1) & (targets == 1)).sum()
        fp = ((preds_bin == 1) & (targets == 0)).sum()
        tn = ((preds_bin == 0) & (targets == 0)).sum()
        fn = ((preds_bin == 0) & (targets == 1)).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
        
        avg_loss = total_loss / len(train_loader)
        
        log_msg = (f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | "
                   f"Val Acc: {acc:.3f} | F1: {f1:.3f} | "
                   f"P: {precision:.2f} R: {recall:.2f}")
        logger.info(log_msg)
        
        # Detailed confusion matrix every 5 epochs or if F1 improves
        if (epoch + 1) % 5 == 0 or f1 > best_f1:
             logger.info(f"   [Matrix] TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")

        if f1 > best_f1:
            best_f1 = f1
            save_path = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
            torch.save(model.state_dict(), save_path)
            logger.info(f"   ⭐ New Best Model Saved! (F1: {f1:.4f})")

    logger.info("="*50)
    logger.info(f"🏆 Training Complete. Best F1: {best_f1:.4f}")
    logger.info(f"💾 Log saved to: {log_file}")

if __name__ == '__main__':
    train()