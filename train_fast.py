"""
⚡ ULTRA-FAST TRAINING with cached YOLO11n features
10x faster than original training! 
WITH CONSOLE LOGGING TO FILE
"""

import torch
from torch.utils.data import random_split
from cached_dataset import CachedFeatureDataset
from accident_detection_model import AccidentDetectionLSTM, AccidentDetectionTransformer
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import time
import sys
from datetime import datetime


# ============================================================================
# Logger Class to write to both console and file
# ============================================================================

class Logger:
    """Logs to both console and file simultaneously"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self. log = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Ensure it writes immediately
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=1.3):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
    
    def forward(self, inputs, targets):
        loss = -self.pos_weight * targets * torch.log(inputs + 1e-7) - (1 - targets) * torch.log(1 - inputs + 1e-7)
        return loss.mean()


def manual_batch_iterator(dataset, batch_size, shuffle=False):
    indices = list(range(len(dataset)))
    
    if shuffle:
        np. random.shuffle(indices)
    
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i: i + batch_size]
        batch_features = []
        batch_labels = []
        
        for idx in batch_indices:
            features, label = dataset[idx]
            batch_features.append(features)
            batch_labels.append(label)
        
        batch_features = torch.stack(batch_features)
        batch_labels = torch.stack(batch_labels)
        
        yield batch_features, batch_labels


def find_best_f1_threshold(y_true, y_scores, min_precision=0.35):
    thresholds = np.arange(0.2, 0.8, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    best_metrics = {}
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        
        if y_pred.sum() == 0:
            continue
            
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if precision >= min_precision and f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {'precision': precision, 'recall':   recall, 'f1':  f1}
    
    return best_threshold, best_metrics


def calculate_metrics(y_true, y_scores, threshold=0.5):
    y_pred = (y_scores >= threshold).astype(int)
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    return precision, recall, f1, cm, y_pred


def train_fast():
    # ========================================================================
    # Setup logging
    # ========================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_log_{timestamp}.txt"
    logger = Logger(log_filename)
    sys.stdout = logger
    
    print("="*80)
    print("⚡ ULTRA-FAST TRAINING WITH YOLO11n CACHED FEATURES")
    print("="*80)
    print(f"📝 Logging to: {log_filename}")
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Config
    BATCH_SIZE = 16
    EPOCHS = 100
    LEARNING_RATE = 0.0001
    DEVICE = torch. device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL_TYPE = 'lstm'  # or 'transformer'
    POS_WEIGHT = 1.3
    EARLY_STOP_PATIENCE = 20
    
    print(f"\n⚙️  Configuration:")
    print(f"   Model: {MODEL_TYPE}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print(f"   Positive Weight: {POS_WEIGHT}")
    print(f"   Device: {DEVICE}")
    print(f"   Early Stop Patience: {EARLY_STOP_PATIENCE}")
    
    # Load cached dataset
    print(f"\n{'='*80}")
    full_dataset = CachedFeatureDataset('feature_cache/yolo11n_features_seq16.pkl')
    
    # Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\n✓ Data split: {len(train_dataset)} train / {len(val_dataset)} val")
    
    # Model
    print(f"\n{'='*80}")
    if MODEL_TYPE == 'lstm':  
        model = AccidentDetectionLSTM(input_dim=128).to(DEVICE)
    else:
        model = AccidentDetectionTransformer(input_dim=128).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Training setup
    criterion = WeightedBCELoss(pos_weight=POS_WEIGHT)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler. ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7, verbose=False)
    
    best_f1 = 0.0
    best_threshold = 0.5
    best_epoch = 0
    epochs_without_improvement = 0
    
    print(f"\n{'='*80}")
    print("🚀 Starting Training...")
    print(f"{'='*80}")
    
    training_start_time = time.time()
    
    # Store history
    history = {
        'epoch': [],
        'train_loss': [],
        'val_f1_05': [],
        'val_f1_opt': [],
        'val_precision':  [],
        'val_recall':  [],
        'threshold': [],
        'lr': []
    }
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (features, labels) in enumerate(manual_batch_iterator(train_dataset, BATCH_SIZE, shuffle=True)):
            features = features.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}:   Loss = {loss.item():.4f}")
        
        train_loss /= train_batches
        
        # Validation
        model.eval()
        val_scores = []
        val_labels_list = []
        
        with torch.no_grad():
            for features, labels in manual_batch_iterator(val_dataset, BATCH_SIZE):
                features = features.to(DEVICE)
                outputs = model(features)
                
                val_scores.extend(outputs.cpu().numpy().flatten())
                val_labels_list.extend(labels.numpy())
        
        val_scores = np.array(val_scores)
        val_labels_list = np.array(val_labels_list)
        
        # Metrics at 0.5
        p_50, r_50, f1_50, cm_50, _ = calculate_metrics(val_labels_list, val_scores, 0.5)
        
        # Optimal threshold
        opt_threshold, opt_metrics = find_best_f1_threshold(val_labels_list, val_scores, min_precision=0.35)
        
        if opt_metrics:  
            f1_opt = opt_metrics['f1']
            p_opt = opt_metrics['precision']
            r_opt = opt_metrics['recall']
            _, _, _, cm_opt, preds_opt = calculate_metrics(val_labels_list, val_scores, opt_threshold)
        else:
            opt_threshold = 0.5
            f1_opt, p_opt, r_opt = f1_50, p_50, r_50
            cm_opt = cm_50
            preds_opt = (val_scores >= 0.5).astype(int)
        
        scheduler.step(f1_opt)
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        
        # Store history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_f1_05'].append(f1_50)
        history['val_f1_opt'].append(f1_opt)
        history['val_precision'].append(p_opt)
        history['val_recall'].append(r_opt)
        history['threshold'].append(opt_threshold)
        history['lr'].append(current_lr)
        
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{EPOCHS} - {epoch_time:.1f}s | LR: {current_lr:.6f}")
        print(f"{'='*50}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"\n📊 Threshold=0.5: P={p_50:.3f} R={r_50:.3f} F1={f1_50:.3f}")
        print(f"🎯 Threshold={opt_threshold:.2f}:   P={p_opt:.3f} R={r_opt:.3f} F1={f1_opt:.3f}")
        print(f"\nConfusion Matrix:\n{cm_opt}")
        print(f"Predicted: {preds_opt. sum():.0f} accidents / {len(preds_opt)-preds_opt.sum():.0f} normal")
        
        # Score distribution
        acc_scores = val_scores[val_labels_list == 1]
        norm_scores = val_scores[val_labels_list == 0]
        separation = abs(acc_scores.mean() - norm_scores.mean())
        print(f"Score separation: {separation:.3f}")
        
        # Save best
        if f1_opt > best_f1:
            best_f1 = f1_opt
            best_threshold = opt_threshold
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_f1':  f1_opt,
                'threshold': opt_threshold,
                'precision': p_opt,
                'recall': r_opt,
                'model_type': MODEL_TYPE,
                'confusion_matrix': cm_opt. tolist(),
            }, 'best_model_fast.pth')
            print(f"✅ Saved best model (F1: {f1_opt:.4f})")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"\n⚠️  Early stopping after {epoch+1} epochs")
            print(f"   No improvement for {EARLY_STOP_PATIENCE} consecutive epochs")
            break
    
    total_time = time.time() - training_start_time
    
    print(f"\n{'='*80}")
    print("✅ TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"⏱️  Total training time:  {total_time/60:.2f} minutes ({total_time:.1f} seconds)")
    print(f"🎯 Best F1 Score: {best_f1:.4f} (achieved at epoch {best_epoch})")
    print(f"🎚️  Optimal Threshold: {best_threshold:.2f}")
    print(f"📊 Total Epochs Run: {len(history['epoch'])}/{EPOCHS}")
    print(f"\n💾 Model saved as:   best_model_fast.pth")
    print(f"📝 Training log saved as: {log_filename}")
    print(f"🕒 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Print summary statistics
    print(f"\n{'='*80}")
    print("📈 TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"Best Results (Epoch {best_epoch}):")
    print(f"  F1 Score: {best_f1:.4f}")
    print(f"  Precision: {history['val_precision'][best_epoch-1]:.4f}")
    print(f"  Recall: {history['val_recall'][best_epoch-1]:.4f}")
    print(f"  Threshold: {best_threshold:.2f}")
    print(f"\nFinal Results (Epoch {len(history['epoch'])}):")
    print(f"  Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"  Val F1 @ 0.5: {history['val_f1_05'][-1]:.4f}")
    print(f"  Val F1 (optimal): {history['val_f1_opt'][-1]:.4f}")
    
    # Save history to CSV
    history_filename = f"training_history_{timestamp}.csv"
    with open(history_filename, 'w') as f:
        f.write("epoch,train_loss,val_f1_05,val_f1_opt,precision,recall,threshold,learning_rate\n")
        for i in range(len(history['epoch'])):
            f.write(f"{history['epoch'][i]},{history['train_loss'][i]:.6f},{history['val_f1_05'][i]:.6f},")
            f.write(f"{history['val_f1_opt'][i]:.6f},{history['val_precision'][i]:.6f},{history['val_recall'][i]:.6f},")
            f.write(f"{history['threshold'][i]:.4f},{history['lr'][i]:.8f}\n")
    
    print(f"\n📊 Training history saved as: {history_filename}")
    print(f"{'='*80}\n")
    
    # Close logger
    sys.stdout = logger.terminal
    logger.close()
    
    print(f"\n✅ All outputs saved to: {log_filename}")


if __name__ == '__main__':
    train_fast()