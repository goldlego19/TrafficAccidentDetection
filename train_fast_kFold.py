"""
Training with K-Fold Cross-Validation + LOGGING
Ensures fair, consistent evaluation
"""

import torch
from torch.utils. data import Subset
from cached_dataset import CachedFeatureDataset
from accident_detection_model import AccidentDetectionLSTM
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import numpy as np
import time
import sys
from datetime import datetime


# ============================================================================
# Logger Class
# ============================================================================

class Logger:
    """Logs to both console and file simultaneously"""
    def __init__(self, filename):
        self.terminal = sys. stdout
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


# ============================================================================
# Training Functions
# ============================================================================

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight=1.3):
        super().__init__()
        self.pos_weight = pos_weight
    
    def forward(self, inputs, targets):
        loss = -self.pos_weight * targets * torch.log(inputs + 1e-7) - (1 - targets) * torch.log(1 - inputs + 1e-7)
        return loss.mean()


def manual_batch_iterator(dataset, batch_size, shuffle=False):
    indices = list(range(len(dataset)))
    if shuffle:
        np. random.shuffle(indices)
    
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_features = []
        batch_labels = []
        
        for idx in batch_indices:
            features, label = dataset[idx]
            batch_features.append(features)
            batch_labels.append(label)
        
        yield torch.stack(batch_features), torch.stack(batch_labels)


def train_one_fold(train_dataset, val_dataset, fold_num, device):
    """Train one fold"""
    print(f"\n{'='*80}")
    print(f"FOLD {fold_num}/5")
    print(f"{'='*80}")
    
    model = AccidentDetectionLSTM(input_dim=128).to(device)
    criterion = WeightedBCELoss(pos_weight=1.3)
    optimizer = torch. optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7, verbose=False
    )
    
    best_f1 = 0.0
    best_epoch = 0
    best_precision = 0.0
    best_recall = 0.0
    best_cm = None
    patience = 0
    
    fold_start = time.time()
    
    for epoch in range(100):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for features, labels in manual_batch_iterator(train_dataset, 16, shuffle=True):
            features = features.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn. utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer. step()
            
            train_loss += loss.item()
            train_batches += 1
        
        train_loss /= train_batches
        
        # Validation
        model.eval()
        val_scores = []
        val_labels_list = []
        
        with torch. no_grad():
            for features, labels in manual_batch_iterator(val_dataset, 16):
                features = features.to(device)
                outputs = model(features)
                val_scores. extend(outputs.cpu().numpy().flatten())
                val_labels_list.extend(labels.numpy())
        
        val_scores = np.array(val_scores)
        val_labels_list = np.array(val_labels_list)
        
        # Metrics
        y_pred = (val_scores >= 0.5).astype(int)
        f1 = f1_score(val_labels_list, y_pred, zero_division=0)
        precision = precision_score(val_labels_list, y_pred, zero_division=0)
        recall = recall_score(val_labels_list, y_pred, zero_division=0)
        cm = confusion_matrix(val_labels_list, y_pred)
        
        scheduler.step(f1)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}:  Loss={train_loss:.4f} | F1={f1:.4f} | P={precision:.4f} | R={recall:.4f} | LR={current_lr:.6f}")
        
        # Track best
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch + 1
            best_cm = cm
            best_precision = precision
            best_recall = recall
            patience = 0
        else:
            patience += 1
        
        if patience >= 20:
            print(f"  ⚠️  Early stopping at epoch {epoch+1} (no improvement for 20 epochs)")
            break
    
    fold_time = time.time() - fold_start
    
    print(f"\n{'='*80}")
    print(f"✅ FOLD {fold_num} COMPLETE (Time: {fold_time:.1f}s)")
    print(f"{'='*80}")
    print(f"Best Results (Epoch {best_epoch}):")
    print(f"  F1 Score:   {best_f1:.4f}")
    print(f"  Precision: {best_precision:.4f}")
    print(f"  Recall:    {best_recall:.4f}")
    print(f"  Confusion Matrix:")
    print(f"{best_cm}")
    
    # Return detailed results
    return {
        'f1': best_f1,
        'precision':  best_precision,
        'recall': best_recall,
        'cm': best_cm,
        'best_epoch': best_epoch,
        'time': fold_time
    }


def train_kfold():
    """5-Fold Cross-Validation with Logging"""
    
    # ========================================================================
    # Setup logging
    # ========================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_kfold_log_{timestamp}.txt"
    logger = Logger(log_filename)
    sys.stdout = logger
    
    print("="*80)
    print("K-FOLD CROSS-VALIDATION TRAINING")
    print("="*80)
    print(f"📝 Logging to: {log_filename}")
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ========================================================================
    # Configuration
    # ========================================================================
    print(f"\n⚙️  Configuration:")
    print(f"   Model:  LSTM (Bidirectional + Attention)")
    print(f"   K-Folds: 5")
    print(f"   Batch Size: 16")
    print(f"   Learning Rate: 0.0001")
    print(f"   Positive Weight: 1.3")
    print(f"   Early Stop Patience: 20")
    print(f"   Max Epochs per Fold: 100")
    
    # ========================================================================
    # Load dataset
    # ========================================================================
    print(f"\n{'='*80}")
    full_dataset = CachedFeatureDataset('feature_cache/yolo11n_features_seq16_sliding3.pkl')
    
    # Get all labels for stratification
    all_labels = np.array([full_dataset.cache[idx]['label'] for idx in full_dataset.indices])
    all_indices = np.array(range(len(full_dataset)))
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Total samples: {len(full_dataset)}")
    print(f"   Accidents: {all_labels.sum():.0f} ({all_labels.mean()*100:.1f}%)")
    print(f"   Normal: {(1-all_labels).sum():.0f} ({(1-all_labels).mean()*100:.1f}%)")
    
    # ========================================================================
    # 5-Fold stratified split
    # ========================================================================
    print(f"\n{'='*80}")
    print("Creating Stratified K-Fold Splits...")
    print(f"{'='*80}")
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    fold_results = []
    overall_start = time.time()
    
    # ========================================================================
    # Train each fold
    # ========================================================================
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_indices, all_labels), 1):
        train_dataset = Subset(full_dataset, train_idx. tolist())
        val_dataset = Subset(full_dataset, val_idx.tolist())
        
        # Count classes in this fold
        train_labels = all_labels[train_idx]
        val_labels = all_labels[val_idx]
        
        print(f"\n{'='*80}")
        print(f"Fold {fold}/5 - Data Split:")
        print(f"{'='*80}")
        print(f"  Train: {len(train_idx)} samples")
        print(f"    - {train_labels.sum():.0f} accidents ({train_labels.mean()*100:.1f}%)")
        print(f"    - {(1-train_labels).sum():.0f} normal ({(1-train_labels).mean()*100:.1f}%)")
        print(f"  Val:    {len(val_idx)} samples")
        print(f"    - {val_labels.sum():.0f} accidents ({val_labels.mean()*100:.1f}%)")
        print(f"    - {(1-val_labels).sum():.0f} normal ({(1-val_labels).mean()*100:.1f}%)")
        
        # Train this fold
        result = train_one_fold(train_dataset, val_dataset, fold, device)
        fold_results.append(result)
    
    # ========================================================================
    # Summary
    # ========================================================================
    total_time = time.time() - overall_start
    
    print(f"\n{'='*80}")
    print("CROSS-VALIDATION COMPLETE!")
    print(f"{'='*80}")
    print(f"⏱️  Total time: {total_time/60:.2f} minutes ({total_time:.1f} seconds)")
    print(f"🕒 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Individual fold results
    print(f"\n{'='*80}")
    print("INDIVIDUAL FOLD RESULTS:")
    print(f"{'='*80}")
    print(f"{'Fold':<6} {'F1 Score':<10} {'Precision':<12} {'Recall':<10} {'Best Epoch':<12} {'Time (s)':<10}")
    print("-" * 80)
    
    for i, result in enumerate(fold_results, 1):
        print(f"{i:<6} {result['f1']:<10.4f} {result['precision']:<12.4f} {result['recall']:<10.4f} {result['best_epoch']:<12} {result['time']:<10.1f}")
    
    # Average results
    avg_f1 = np.mean([r['f1'] for r in fold_results])
    avg_prec = np.mean([r['precision'] for r in fold_results])
    avg_rec = np.mean([r['recall'] for r in fold_results])
    std_f1 = np.std([r['f1'] for r in fold_results])
    std_prec = np.std([r['precision'] for r in fold_results])
    std_rec = np.std([r['recall'] for r in fold_results])
    
    print(f"\n{'='*80}")
    print("📊 AVERAGE PERFORMANCE ACROSS ALL FOLDS:")
    print(f"{'='*80}")
    print(f"  F1 Score:   {avg_f1:.4f} ± {std_f1:.4f}")
    print(f"  Precision: {avg_prec:.4f} ± {std_prec:.4f}")
    print(f"  Recall:    {avg_rec:.4f} ± {std_rec:.4f}")
    
    # Aggregate confusion matrix
    total_cm = sum([r['cm'] for r in fold_results])
    print(f"\n  Aggregate Confusion Matrix (all folds):")
    print(f"{total_cm}")
    
    # Calculate aggregate metrics from confusion matrix
    tn, fp, fn, tp = total_cm.ravel()
    agg_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    agg_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    agg_f1 = 2 * (agg_precision * agg_recall) / (agg_precision + agg_recall) if (agg_precision + agg_recall) > 0 else 0
    
    print(f"\n  Aggregate Metrics:")
    print(f"    Precision: {agg_precision:.4f}")
    print(f"    Recall:    {agg_recall:.4f}")
    print(f"    F1:         {agg_f1:.4f}")
    
    print(f"\n{'='*80}")
    print(f"💾 Complete log saved to: {log_filename}")
    print(f"{'='*80}\n")
    
    # Save results to CSV
    csv_filename = f"kfold_results_{timestamp}.csv"
    with open(csv_filename, 'w') as f:
        f.write("fold,f1,precision,recall,best_epoch,time_seconds\n")
        for i, result in enumerate(fold_results, 1):
            f.write(f"{i},{result['f1']:.6f},{result['precision']:.6f},{result['recall']:.6f},{result['best_epoch']},{result['time']:.2f}\n")
        f.write(f"average,{avg_f1:.6f},{avg_prec:.6f},{avg_rec:.6f},-,-\n")
        f.write(f"std_dev,{std_f1:.6f},{std_prec:.6f},{std_rec:.6f},-,-\n")
    
    print(f"📊 Results also saved to: {csv_filename}\n")
    
    # Close logger
    sys.stdout = logger.terminal
    logger.close()
    
    print(f"✅ All outputs saved to: {log_filename}")
    
    return fold_results, avg_f1, avg_prec, avg_rec


if __name__ == '__main__':
    results, avg_f1, avg_prec, avg_rec = train_kfold()
    
    print(f"\n{'='*80}")
    print("🎯 FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"Average F1 Score:   {avg_f1:.4f}")
    print(f"Average Precision: {avg_prec:.4f}")
    print(f"Average Recall:    {avg_rec:.4f}")
    print(f"{'='*80}\n")