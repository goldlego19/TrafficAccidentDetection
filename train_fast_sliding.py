"""
⚡ YOLO11n + LSTM Accident Detection Training
Optimized for 80%+ precision with sliding window augmentation
"""

import torch
from torch.utils.data import random_split
from cached_dataset_sliding import CachedFeatureDatasetSliding
from accident_detection_model import AccidentDetectionLSTM
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import time
import sys
from datetime import datetime
from pathlib import Path


class Logger: 
    """Dual output:  console + file"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


class WeightedBCELoss(nn.Module):
    """BCE Loss with class weighting"""
    def __init__(self, pos_weight=1.3):
        super().__init__()
        self.pos_weight = pos_weight
    
    def forward(self, inputs, targets):
        loss = -self.pos_weight * targets * torch.log(inputs + 1e-7) - \
               (1 - targets) * torch.log(1 - inputs + 1e-7)
        return loss.mean()


def manual_batch_iterator(dataset, batch_size, shuffle=False):
    """Manual batching"""
    indices = list(range(len(dataset)))
    if shuffle:
        np.random.shuffle(indices)
    
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_features = []
        batch_labels = []
        
        for idx in batch_indices:
            features, label = dataset[idx]
            batch_features.append(features)
            batch_labels.append(label)
        
        yield torch. stack(batch_features), torch.stack(batch_labels)


def find_best_f1_threshold(y_true, y_scores, min_precision=0.35):
    """Find optimal threshold for F1"""
    thresholds = np.arange(0.2, 0.8, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    best_metrics = {}
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        
        if y_pred. sum() == 0:
            continue
            
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if precision >= min_precision and f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {'precision': precision, 'recall': recall, 'f1': f1}
    
    return best_threshold, best_metrics


def calculate_metrics(y_true, y_scores, threshold=0.5):
    """Calculate all metrics at threshold"""
    y_pred = (y_scores >= threshold).astype(int)
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    return precision, recall, f1, cm, y_pred


def train_yolo11n():
    """Main training function for YOLO11n"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_yolo11n_{timestamp}.txt"
    logger = Logger(log_filename)
    sys.stdout = logger
    
    print("="*80)
    print("⚡ YOLO11n + LSTM ACCIDENT DETECTION TRAINING")
    print("="*80)
    print(f"📝 Logging to: {log_filename}")
    print(f"🕒 Started at: {datetime. now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    BATCH_SIZE = 16
    EPOCHS = 100
    LEARNING_RATE = 0.0001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    POS_WEIGHT = 1.3
    EARLY_STOP_PATIENCE = 20
    CACHE_FILE = 'feature_cache/yolo11n_features_seq16_sliding3.pkl'
    
    print(f"\n⚙️  Configuration:")
    print(f"   YOLO Backbone: YOLO11n")
    print(f"   Model: LSTM (Bidirectional + Attention)")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print(f"   Positive Weight: {POS_WEIGHT}")
    print(f"   Device: {DEVICE}")
    print(f"   Early Stop Patience: {EARLY_STOP_PATIENCE}")
    print(f"   Data:  Sliding Window (3 clips per video)")
    
    # Check cache exists
    print(f"\n{'='*80}")
    if not Path(CACHE_FILE).exists():
        print(f"❌ ERROR: Cache file not found:  {CACHE_FILE}")
        print(f"   Please run:  python cache_yolo_features_sliding.py")
        sys.stdout = logger.terminal
        logger. close()
        return
    
    # Load dataset
    full_dataset = CachedFeatureDatasetSliding(CACHE_FILE)
    
    # Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\n✓ Data split:  {len(train_dataset)} train / {len(val_dataset)} val")
    
    # Model
    print(f"\n{'='*80}")
    model = AccidentDetectionLSTM(input_dim=128).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Training setup
    criterion = WeightedBCELoss(pos_weight=POS_WEIGHT)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7, verbose=False
    )
    
    best_val_f1 = 0.0
    best_threshold = 0.5
    best_epoch = 0
    epochs_without_improvement = 0
    
    print(f"\n{'='*80}")
    print("🚀 Starting Training...")
    print(f"{'='*80}")
    
    training_history = {
        'epoch': [], 'train_loss': [], 'val_f1_05': [], 'val_f1_opt': [],
        'val_precision':  [], 'val_recall': [], 'threshold': [], 'lr': []
    }
    
    start_time = time.time()
    
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
            
            if (batch_idx + 1) % 20 == 0:
                print(f"  Batch {batch_idx + 1}:  Loss = {loss.item():.4f}")
        
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
        
        # Metrics
        p_50, r_50, f1_50, cm_50, _ = calculate_metrics(val_labels_list, val_scores, 0.5)
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
        training_history['epoch'].append(epoch + 1)
        training_history['train_loss'].append(train_loss)
        training_history['val_f1_05']. append(f1_50)
        training_history['val_f1_opt'].append(f1_opt)
        training_history['val_precision'].append(p_opt)
        training_history['val_recall'].append(r_opt)
        training_history['threshold'].append(opt_threshold)
        training_history['lr'].append(current_lr)
        
        # Print progress
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{EPOCHS} - {epoch_time:.1f}s | LR: {current_lr:.6f}")
        print(f"{'='*50}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"\n📊 Threshold=0.5: P={p_50:.3f} R={r_50:.3f} F1={f1_50:.3f}")
        print(f"🎯 Threshold={opt_threshold:.2f}:  P={p_opt:.3f} R={r_opt:.3f} F1={f1_opt:.3f}")
        print(f"\nConfusion Matrix:\n{cm_opt}")
        print(f"Predicted: {preds_opt.sum():.0f} accidents / {len(preds_opt)-preds_opt.sum():.0f} normal")
        
        # Score separation
        acc_scores = val_scores[val_labels_list == 1]
        norm_scores = val_scores[val_labels_list == 0]
        if len(acc_scores) > 0 and len(norm_scores) > 0:
            separation = abs(acc_scores. mean() - norm_scores.mean())
            print(f"Score separation: {separation:.3f}")
        
        # Save best
        if f1_opt > best_val_f1:
            best_val_f1 = f1_opt
            best_threshold = opt_threshold
            best_epoch = epoch + 1
            
            torch.save({
                'epoch':  epoch,
                'model_state_dict': model.state_dict(),
                'val_f1':  f1_opt,
                'threshold': opt_threshold,
                'precision': p_opt,
                'recall': r_opt,
                'yolo_backbone': 'yolo11n',
                'cache_file': CACHE_FILE,
            }, 'best_model_yolo11n.pth')
            print(f"✅ Saved best model (F1: {f1_opt:.4f})")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= EARLY_STOP_PATIENCE: 
            print(f"\n⚠️ Early stopping after {epoch+1} epochs")
            break
    
    total_time = time.time() - start_time
    
    # Final summary
    print(f"\n{'='*80}")
    print("✅ TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"⏱️  Total time: {total_time/60:.2f} minutes ({total_time:.1f} seconds)")
    print(f"🎯 Best F1: {best_val_f1:.4f} (epoch {best_epoch})")
    print(f"🎚️  Best threshold: {best_threshold:.2f}")
    
    # Test 80% precision threshold
    print(f"\n{'='*80}")
    print("🔍 TESTING FOR 80% PRECISION...")
    print(f"{'='*80}")
    
    # Get final validation predictions
    model.eval()
    final_scores = []
    final_labels = []
    
    with torch. no_grad():
        for features, labels in manual_batch_iterator(val_dataset, BATCH_SIZE):
            features = features. to(DEVICE)
            outputs = model(features)
            final_scores.extend(outputs.cpu().numpy().flatten())
            final_labels.extend(labels.numpy())
    
    final_scores = np.array(final_scores)
    final_labels = np.array(final_labels)
    
    # Find 80% precision threshold
    found_80 = False
    for thresh in np.arange(0.50, 0.90, 0.05):
        y_pred = (final_scores >= thresh).astype(int)
        if y_pred.sum() > 0:
            p = precision_score(final_labels, y_pred, zero_division=0)
            r = recall_score(final_labels, y_pred, zero_division=0)
            f1 = f1_score(final_labels, y_pred, zero_division=0)
            
            if p >= 0.80:
                print(f"\n✅ 80% PRECISION ACHIEVED!")
                print(f"   Threshold: {thresh:.2f}")
                print(f"   Precision: {p:.4f}")
                print(f"   Recall: {r:.4f}")
                print(f"   F1: {f1:.4f}")
                found_80 = True
                break
    
    if not found_80:
        print(f"\n⚠️  80% precision not achieved in this run")
        print(f"   (May vary with different random splits)")
    
    # Save files
    print(f"\n{'='*80}")
    print(f"💾 Saved Files:")
    print(f"{'='*80}")
    print(f"   Model: best_model_yolo11n.pth")
    print(f"   Log: {log_filename}")
    
    history_file = f'training_history_yolo11n_{timestamp}.npy'
    np.save(history_file, training_history)
    print(f"   History:  {history_file}")
    
    print(f"\n🕒 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    sys.stdout = logger.terminal
    logger.close()
    
    print(f"\n✅ Training complete! Check {log_filename} for full details.")


if __name__ == '__main__':
    train_yolo11n()