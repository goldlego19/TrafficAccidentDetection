"""
FINAL FIXED Training script with dynamic threshold optimization
"""

import torch
from torch.utils.data import random_split
import torchvision.transforms as transforms
from accident_detection_model import (
    TADFrameDataset,
    AccidentDetectionPipeline
)
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_curve
import time


class WeightedBCELoss(nn.Module):
    """Weighted BCE Loss for class imbalance"""
    def __init__(self, pos_weight=2.27):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight
    
    def forward(self, inputs, targets):
        # Manual weighted BCE
        loss = -self.pos_weight * targets * torch.log(inputs + 1e-7) - (1 - targets) * torch.log(1 - inputs + 1e-7)
        return loss.mean()


def get_transforms(augment=False):
    if augment:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms. Resize((224, 224)),
            transforms. RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def manual_batch_iterator(dataset, batch_size, shuffle=False):
    indices = list(range(len(dataset)))
    
    if shuffle:
        np. random.shuffle(indices)
    
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_frames = []
        batch_labels = []
        
        for idx in batch_indices:
            frames, label = dataset[idx]
            batch_frames.append(frames)
            batch_labels.append(label)
        
        batch_frames = torch.stack(batch_frames)
        batch_labels = torch.stack(batch_labels)
        
        yield batch_frames, batch_labels


def find_optimal_threshold(y_true, y_scores):
    """Find optimal threshold that maximizes F1 score"""
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold, best_f1


def calculate_metrics_with_threshold(y_true, y_scores, threshold=0.5):
    """Calculate metrics with specific threshold"""
    y_pred = (y_scores >= threshold).astype(int)
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    return precision, recall, f1, cm, y_pred


def train_on_tad_dataset():
    print("="*80)
    print("FINAL TRAINING - Traffic Accident Detection")
    print("="*80)
    
    # Configuration
    SEQUENCE_LENGTH = 16
    BATCH_SIZE = 8
    EPOCHS = 100
    LEARNING_RATE = 0.0001
    FEATURE_DIM = 128
    EARLY_STOP_PATIENCE = 20
    MODEL_TYPE = 'transformer'
    
    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_TYPE}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Early Stop Patience: {EARLY_STOP_PATIENCE}")
    
    # Create datasets
    print("\n" + "="*80)
    train_transform = get_transforms(augment=True)
    val_transform = get_transforms(augment=False)
    
    full_dataset = TADFrameDataset(
        root_dir='TAD/frames',
        sequence_length=SEQUENCE_LENGTH,
        transform=None,
        augment=False
    )
    
    if len(full_dataset) == 0:
        print("ERROR:  No valid sequences found!")
        return
    
    # Split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_dataset.dataset.transform = train_transform
    train_dataset.dataset.augment = True
    val_dataset.dataset.transform = val_transform
    
    # Count classes
    train_accidents = sum([train_dataset.dataset.labels[idx] for idx in train_dataset. indices])
    val_accidents = sum([val_dataset.dataset. labels[idx] for idx in val_dataset.indices])
    
    print(f"\n✓ Data split:")
    print(f"  Train: {train_accidents} accidents / {len(train_dataset)-train_accidents} normal")
    print(f"  Val: {val_accidents} accidents / {len(val_dataset)-val_accidents} normal")
    
    # Class weight
    pos_weight = (len(train_dataset) - train_accidents) / train_accidents
    print(f"  Positive class weight: {pos_weight:.2f}")
    
    # Initialize pipeline
    print("\n" + "="*80)
    pipeline = AccidentDetectionPipeline(
        yolo_model='yolo11n.pt',
        temporal_model=MODEL_TYPE,
        input_dim=FEATURE_DIM,
        device='cuda'
    )
    
    # Weighted BCE Loss
    criterion = WeightedBCELoss(pos_weight=pos_weight)
    print(f"Using Weighted BCE Loss (pos_weight={pos_weight:.2f})")
    
    optimizer = torch.optim. AdamW(
        pipeline.model.parameters(), 
        lr=LEARNING_RATE,
        weight_decay=0.01
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7, verbose=True
    )
    
    # Tracking
    best_val_f1 = 0.0
    best_threshold = 0.5
    epochs_without_improvement = 0
    
    print("\n" + "="*80)
    print("Starting Training...")
    print("="*80)
    
    training_history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': [],
        'val_precision': [], 'val_recall': [], 'thresholds': []
    }
    
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 50)
        
        # Training
        pipeline.model.train()
        train_loss = 0.0
        train_scores = []
        train_labels_list = []
        train_batches = 0
        
        for batch_idx, (frames, labels) in enumerate(manual_batch_iterator(train_dataset, BATCH_SIZE, shuffle=True)):
            features = pipeline.extract_features_from_frames(frames)
            labels = labels.to(pipeline.device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = pipeline.model(features)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pipeline.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_scores.extend(outputs.detach().cpu().numpy().flatten())
            train_labels_list.extend(labels.cpu().numpy().flatten())
            train_batches += 1
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}:  Loss = {loss.item():.4f}")
        
        train_loss /= train_batches
        train_scores = np.array(train_scores)
        train_labels_list = np.array(train_labels_list)
        
        # Find optimal threshold on training data
        train_threshold, _ = find_optimal_threshold(train_labels_list, train_scores)
        train_precision, train_recall, train_f1, train_cm, _ = calculate_metrics_with_threshold(
            train_labels_list, train_scores, train_threshold
        )
        train_acc = np.mean((train_scores >= train_threshold) == train_labels_list)
        
        # Validation
        pipeline.model.eval()
        val_loss = 0.0
        val_scores = []
        val_labels_list = []
        val_batches = 0
        
        with torch.no_grad():
            for frames, labels in manual_batch_iterator(val_dataset, BATCH_SIZE, shuffle=False):
                features = pipeline.extract_features_from_frames(frames)
                labels_tensor = labels.to(pipeline.device).unsqueeze(1)
                
                outputs = pipeline.model(features)
                loss = criterion(outputs, labels_tensor)
                
                val_loss += loss.item()
                val_scores.extend(outputs.cpu().numpy().flatten())
                val_labels_list.extend(labels_tensor.cpu().numpy().flatten())
                val_batches += 1
        
        val_loss /= val_batches
        val_scores = np.array(val_scores)
        val_labels_list = np.array(val_labels_list)
        
        # Find optimal threshold on validation data
        val_threshold, val_f1_optimal = find_optimal_threshold(val_labels_list, val_scores)
        
        # Calculate metrics with optimal threshold
        precision, recall, f1, cm, val_preds = calculate_metrics_with_threshold(
            val_labels_list, val_scores, val_threshold
        )
        val_acc = np.mean(val_preds == val_labels_list)
        
        # Also show metrics with 0.5 threshold for comparison
        precision_50, recall_50, f1_50, cm_50, val_preds_50 = calculate_metrics_with_threshold(
            val_labels_list, val_scores, 0.5
        )
        
        # Scheduler
        scheduler.step(f1)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['train_f1'].append(train_f1)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        training_history['val_f1'].append(f1)
        training_history['val_precision'].append(precision)
        training_history['val_recall'].append(recall)
        training_history['thresholds'].append(val_threshold)
        
        # Print metrics
        epoch_time = time.time() - epoch_start
        print(f"\n  Time: {epoch_time:.1f}s | LR: {current_lr:.6f}")
        print(f"  TRAIN - Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f} | Threshold: {train_threshold:.2f}")
        print(f"  VAL   - Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {f1:.4f} | Threshold: {val_threshold:.2f}")
        print(f"  Val Precision: {precision:.4f} | Recall: {recall:.4f}")
        print(f"  Confusion Matrix (threshold={val_threshold:.2f}):\n{cm}")
        print(f"  With threshold=0.5: Precision:  {precision_50:.4f} | Recall: {recall_50:.4f} | F1: {f1_50:.4f}")
        
        # Score distribution
        print(f"  Prediction scores - Min: {val_scores.min():.3f}, Max: {val_scores.max():.3f}, Mean: {val_scores.mean():.3f}")
        
        # Save best model
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_threshold = val_threshold
            torch.save({
                'epoch': epoch,
                'model_state_dict': pipeline.model. state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_f1':  f1,
                'val_precision': precision,
                'val_recall': recall,
                'threshold': val_threshold,
            }, 'best_model.pth')
            print(f"  ✓ Saved best model (F1: {f1:.4f}, Threshold: {val_threshold:.2f})")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= EARLY_STOP_PATIENCE:
            print(f"\n{'='*80}")
            print(f"Early stopping after {epoch+1} epochs")
            break
    
    # Complete
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print("Training Completed!")
    print(f"{'='*80}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"Optimal threshold:  {best_threshold:.2f}")
    
    # Load best
    checkpoint = torch.load('best_model.pth')
    pipeline.model.load_state_dict(checkpoint['model_state_dict'])
    
    # Save
    torch.save({
        'model_state_dict': pipeline.model.state_dict(),
        'threshold': best_threshold,
        'val_f1': best_val_f1,
    }, f'accident_detector_{MODEL_TYPE}_final.pth')
    
    print(f"\nModel saved with optimal threshold: {best_threshold:.2f}")
    np.save('training_history.npy', training_history)
    
    return pipeline, training_history, best_threshold


if __name__ == '__main__':
    pipeline, history, threshold = train_on_tad_dataset()
    print(f"\n🎯 Use threshold {threshold:.2f} for inference!")