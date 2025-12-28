"""
Train 5-model ensemble using K-Fold Cross-Validation - FIXED
Each model trained on different data split for diversity
"""

import torch
from torch.utils.data import Subset
from cached_dataset_sliding import CachedFeatureDatasetSliding
from accident_detection_model import AccidentDetectionLSTM
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import numpy as np
import time
import sys
from datetime import datetime


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self. log = open(filename, 'w', encoding='utf-8')
    
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
    def __init__(self, pos_weight=1.3):
        super().__init__()
        self.pos_weight = pos_weight
    
    def forward(self, inputs, targets):
        loss = -self.pos_weight * targets * torch.log(inputs + 1e-7) - (1 - targets) * torch.log(1 - inputs + 1e-7)
        return loss.mean()


def manual_batch_iterator(dataset, batch_size, shuffle=False):
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
        
        yield torch.stack(batch_features), torch.stack(batch_labels)


def train_one_model(train_dataset, val_dataset, fold_num, device, max_epochs=50):
    """Train one model in the ensemble"""
    print(f"\n{'='*80}")
    print(f"TRAINING MODEL {fold_num}/5")
    print(f"{'='*80}")
    
    model = AccidentDetectionLSTM(input_dim=128).to(device)
    criterion = WeightedBCELoss(pos_weight=1.3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler. ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7, verbose=False
    )
    
    best_f1 = 0.0
    best_epoch = 0
    best_model_state = None
    patience = 0
    
    fold_start = time.time()
    
    for epoch in range(max_epochs):
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        train_loss /= train_batches
        
        # Validation
        model.eval()
        val_scores = []
        val_labels_list = []
        
        with torch.no_grad():
            for features, labels in manual_batch_iterator(val_dataset, 16):
                features = features.to(device)
                outputs = model(features)
                val_scores.extend(outputs.cpu().numpy().flatten())
                val_labels_list.extend(labels.numpy())
        
        val_scores = np.array(val_scores)
        val_labels_list = np.array(val_labels_list)
        
        # Metrics at threshold=0.5
        y_pred = (val_scores >= 0.5).astype(int)
        f1 = f1_score(val_labels_list, y_pred, zero_division=0)
        precision = precision_score(val_labels_list, y_pred, zero_division=0)
        recall = recall_score(val_labels_list, y_pred, zero_division=0)
        
        scheduler.step(f1)
        current_lr = optimizer.param_groups[0]['lr']
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}:  Loss={train_loss:.4f} | F1={f1:.4f} | P={precision:.4f} | R={recall:.4f} | LR={current_lr:.6f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
            patience = 0
        else:
            patience += 1
        
        if patience >= 15:
            print(f"  ⚠️  Early stopping at epoch {epoch+1}")
            break
    
    fold_time = time.time() - fold_start
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    val_scores = []
    val_labels_list = []
    
    with torch.no_grad():
        for features, labels in manual_batch_iterator(val_dataset, 16):
            features = features.to(device)
            outputs = model(features)
            val_scores.extend(outputs.cpu().numpy().flatten())
            val_labels_list.extend(labels.numpy())
    
    val_scores = np.array(val_scores)
    val_labels_list = np.array(val_labels_list)
    
    y_pred = (val_scores >= 0.5).astype(int)
    final_f1 = f1_score(val_labels_list, y_pred, zero_division=0)
    final_precision = precision_score(val_labels_list, y_pred, zero_division=0)
    final_recall = recall_score(val_labels_list, y_pred, zero_division=0)
    final_cm = confusion_matrix(val_labels_list, y_pred)
    
    print(f"\n✅ Model {fold_num} Complete (Time: {fold_time:.1f}s)")
    print(f"   Best Epoch: {best_epoch}")
    print(f"   F1: {final_f1:.4f} | Precision: {final_precision:.4f} | Recall: {final_recall:.4f}")
    print(f"   Confusion Matrix:\n{final_cm}")
    
    return {
        'model': model,
        'f1': final_f1,
        'precision': final_precision,
        'recall':  final_recall,
        'cm': final_cm,
        'val_scores': val_scores,
        'val_labels':  val_labels_list,
        'best_epoch': best_epoch,
        'time': fold_time
    }


def train_ensemble():
    """Train 5-model ensemble with K-Fold"""
    
    timestamp = datetime. now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"training_ensemble_log_{timestamp}.txt"
    logger = Logger(log_filename)
    sys.stdout = logger
    
    print("="*80)
    print("🎯 ENSEMBLE TRAINING (5 Models via K-Fold)")
    print("="*80)
    print(f"📝 Logging to: {log_filename}")
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n⚙️  Configuration:")
    print(f"   Ensemble Size: 5 models")
    print(f"   Training Strategy: K-Fold Cross-Validation")
    print(f"   Data:  Sliding Window (1,341 samples)")
    print(f"   Expected Time:  ~2 minutes")
    
    # Load dataset
    print(f"\n{'='*80}")
    full_dataset = CachedFeatureDatasetSliding('feature_cache/yolo11n_features_seq16_sliding3.pkl')
    
    all_labels = np.array([full_dataset.cache[idx]['label'] for idx in full_dataset.indices])
    all_indices = np.array(range(len(full_dataset)))
    
    print(f"\n📊 Dataset: {len(full_dataset)} samples")
    
    # 5-Fold stratified split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models = []
    overall_start = time.time()
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_indices, all_labels), 1):
        train_dataset = Subset(full_dataset, train_idx. tolist())
        val_dataset = Subset(full_dataset, val_idx.tolist())
        
        train_labels = all_labels[train_idx]
        val_labels = all_labels[val_idx]
        
        print(f"\n{'='*80}")
        print(f"Fold {fold}/5:")
        print(f"  Train: {len(train_idx)} ({train_labels.sum():.0f} accidents)")
        print(f"  Val:   {len(val_idx)} ({val_labels.sum():.0f} accidents)")
        
        result = train_one_model(train_dataset, val_dataset, fold, device, max_epochs=50)
        models.append(result)
        
        # Save individual model
        torch.save({
            'model_state_dict': result['model'].state_dict(),
            'f1': result['f1'],
            'precision': result['precision'],
            'recall': result['recall'],
            'fold': fold,
        }, f'ensemble_model_{fold}.pth')
        print(f"   💾 Saved:  ensemble_model_{fold}.pth")
    
    total_time = time.time() - overall_start
    
    # Ensemble evaluation
    print(f"\n{'='*80}")
    print("🎯 ENSEMBLE EVALUATION")
    print(f"{'='*80}")
    
    print(f"\n📊 Individual Model Performance:")
    print(f"{'Model':<8} {'F1':<8} {'Precision':<12} {'Recall':<10} {'Time (s)':<10}")
    print("-"*60)
    
    for i, model_result in enumerate(models, 1):
        print(f"{i:<8} {model_result['f1']:<8.4f} {model_result['precision']:<12.4f} {model_result['recall']: <10.4f} {model_result['time']: <10.1f}")
    
    avg_f1 = np.mean([m['f1'] for m in models])
    avg_prec = np.mean([m['precision'] for m in models])
    avg_rec = np.mean([m['recall'] for m in models])
    std_f1 = np.std([m['f1'] for m in models])
    
    print(f"\nAverage:  {avg_f1:<8.4f} {avg_prec:<12.4f} {avg_rec:<10.4f}")
    print(f"Std Dev: {std_f1:<8.4f}")
    
    # Collect all validation predictions (concatenate across folds) - FIXED
    print(f"\n📊 Combining predictions from all folds...")
    ensemble_scores = []
    ensemble_labels = []
    
    for model_result in models:
        # Use extend to concatenate lists (not numpy arrays)
        ensemble_scores. extend(model_result['val_scores']. tolist())
        ensemble_labels.extend(model_result['val_labels']. tolist())
    
    # Now convert to numpy
    ensemble_scores = np.array(ensemble_scores)
    ensemble_labels = np.array(ensemble_labels)
    
    print(f"Total validation samples: {len(ensemble_scores)}")
    
    # Aggregate confusion matrix (all individual models combined)
    total_cm = sum([m['cm'] for m in models])
    print(f"\nAggregate Confusion Matrix (individual models):")
    print(f"{total_cm}")
    
    # Test ensemble at different thresholds
    print(f"\n{'='*80}")
    print("🎯 ENSEMBLE THRESHOLD ANALYSIS")
    print(f"{'='*80}")
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<10} {'F1':<8} {'TP':<6} {'FP':<6} {'Status':<15}")
    print("-"*80)
    
    thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    best_results = {}
    
    for threshold in thresholds:
        y_pred = (ensemble_scores >= threshold).astype(int)
        
        if y_pred.sum() == 0:
            continue
        
        precision = precision_score(ensemble_labels, y_pred, zero_division=0)
        recall = recall_score(ensemble_labels, y_pred, zero_division=0)
        f1 = f1_score(ensemble_labels, y_pred, zero_division=0)
        cm = confusion_matrix(ensemble_labels, y_pred)
        
        tn, fp, fn, tp = cm.ravel()
        
        status = "✅ 80%+ PRECISION" if precision >= 0.80 else ""
        
        print(f"{threshold:<12.2f} {precision:<12.4f} {recall:<10.4f} {f1:<8.4f} {tp:<6} {fp:<6} {status: <15}")
        
        best_results[threshold] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cm': cm
        }
    
    # Find best F1
    best_f1_thresh = max(best_results.keys(), key=lambda k: best_results[k]['f1'])
    
    # Find 80% precision threshold
    precision_80_thresh = None
    for thresh in sorted(best_results.keys()):
        if best_results[thresh]['precision'] >= 0.80:
            precision_80_thresh = thresh
            break
    
    print(f"\n{'='*80}")
    print("📊 ENSEMBLE SUMMARY")
    print(f"{'='*80}")
    print(f"⏱️  Total Training Time: {total_time/60:.2f} minutes")
    print(f"📈 Individual Models Average: F1={avg_f1:.4f} ± {std_f1:.4f}, P={avg_prec:.4f}, R={avg_rec:.4f}")
    
    print(f"\n🎯 Best F1 Score (Threshold={best_f1_thresh}):")
    print(f"   F1:        {best_results[best_f1_thresh]['f1']:.4f}")
    print(f"   Precision: {best_results[best_f1_thresh]['precision']:.4f}")
    print(f"   Recall:    {best_results[best_f1_thresh]['recall']:.4f}")
    print(f"   Confusion Matrix:\n{best_results[best_f1_thresh]['cm']}")
    
    tn, fp, fn, tp = best_results[best_f1_thresh]['cm'].ravel()
    print(f"\n   📈 At Best F1:")
    print(f"      Detected: {tp}/{tp+fn} accidents ({tp/(tp+fn)*100:.1f}%)")
    print(f"      False Alarms: {fp}")
    print(f"      Correctly Identified Normal: {tn}/{tn+fp} ({tn/(tn+fp)*100:.1f}%)")
    
    if precision_80_thresh:
        print(f"\n✅ 80% PRECISION ACHIEVED (Threshold={precision_80_thresh}):")
        print(f"   Precision:  {best_results[precision_80_thresh]['precision']:.4f}")
        print(f"   Recall:    {best_results[precision_80_thresh]['recall']:.4f}")
        print(f"   F1:        {best_results[precision_80_thresh]['f1']:.4f}")
        print(f"   Confusion Matrix:\n{best_results[precision_80_thresh]['cm']}")
        
        tn, fp, fn, tp = best_results[precision_80_thresh]['cm'].ravel()
        print(f"\n   📈 At 80% Precision:")
        print(f"      Detected: {tp}/{tp+fn} accidents ({tp/(tp+fn)*100:.1f}%)")
        print(f"      False Alarms:  {fp} (out of {tn+fp} normal videos)")
        print(f"      Missed:  {fn} accidents")
        
        # Save ensemble info with 80% precision threshold
        ensemble_info = {
            'num_models': len(models),
            'best_f1_threshold': best_f1_thresh,
            'precision_80_threshold': precision_80_thresh,
            'results': best_results,
        }
        torch.save(ensemble_info, 'ensemble_info.pth')
        print(f"\n   💾 Ensemble info saved:  ensemble_info.pth")
    else:
        print(f"\n⚠️  80% precision not achieved")
        max_prec = max([r['precision'] for r in best_results.values()])
        max_prec_thresh = [t for t, r in best_results.items() if r['precision'] == max_prec][0]
        print(f"   Maximum precision:  {max_prec:.4f} at threshold {max_prec_thresh:.2f}")
    
    print(f"\n💾 Saved Models:")
    for i in range(1, 6):
        print(f"   - ensemble_model_{i}.pth")
    
    print(f"\n📝 Log saved:  {log_filename}")
    print(f"🕒 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    sys.stdout = logger.terminal
    logger.close()
    
    return models, best_results


if __name__ == '__main__':
    models, results = train_ensemble()