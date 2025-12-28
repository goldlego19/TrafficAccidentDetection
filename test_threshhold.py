"""
Complete analysis of YOLO11n model performance
Tests all thresholds to find optimal operating points
"""

import torch
import numpy as np
from cached_dataset_sliding import CachedFeatureDatasetSliding
from accident_detection_model import AccidentDetectionLSTM
from torch.utils.data import random_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt


def analyze_model():
    print("="*80)
    print("🎯 YOLO11n MODEL COMPLETE ANALYSIS")
    print("="*80)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AccidentDetectionLSTM(input_dim=128).to(device)
    
    checkpoint = torch.load('best_model_yolo11n.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"\n✅ Loaded model:")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Val F1: {checkpoint['val_f1']:.4f}")
    print(f"   Trained threshold: {checkpoint['threshold']:.2f}")
    print(f"   Precision: {checkpoint['precision']:.4f}")
    print(f"   Recall: {checkpoint['recall']:.4f}")
    
    # Load validation data
    full_dataset = CachedFeatureDatasetSliding('feature_cache/yolo11n_features_seq16_sliding3.pkl')
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    _, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\nValidation set:  {len(val_dataset)} samples")
    
    # Get predictions
    print("\nGenerating predictions...")
    val_scores = []
    val_labels = []
    
    with torch.no_grad():
        for i in range(len(val_dataset)):
            features, label = val_dataset[i]
            features = features.unsqueeze(0).to(device)
            output = model(features)
            val_scores.append(output.item())
            val_labels.append(label. item())
    
    val_scores = np.array(val_scores)
    val_labels = np.array(val_labels)
    
    num_accidents = val_labels.sum()
    num_normal = len(val_labels) - num_accidents
    
    print(f"  - {num_accidents:.0f} accident clips")
    print(f"  - {num_normal:.0f} normal clips")
    
    # Comprehensive threshold analysis
    print(f"\n{'='*80}")
    print("📊 COMPREHENSIVE THRESHOLD ANALYSIS")
    print(f"{'='*80}")
    print(f"{'Threshold':<12} {'Precision':<12} {'Recall':<10} {'F1':<8} {'TP':<6} {'FP':<6} {'FN':<6} {'TN':<6} {'Notes': <20}")
    print("-"*100)
    
    results = []
    thresholds = np.arange(0.15, 0.90, 0.05)
    
    for threshold in thresholds:
        y_pred = (val_scores >= threshold).astype(int)
        
        if y_pred.sum() == 0:
            continue
        
        precision = precision_score(val_labels, y_pred, zero_division=0)
        recall = recall_score(val_labels, y_pred, zero_division=0)
        f1 = f1_score(val_labels, y_pred, zero_division=0)
        cm = confusion_matrix(val_labels, y_pred)
        
        tn, fp, fn, tp = cm.ravel()
        
        # Determine notes
        notes = []
        if f1 >= 0.56:
            notes.append("Best F1")
        if precision >= 0.80:
            notes.append("✅ 80%+ Prec")
        if recall >= 0.80:
            notes. append("High Recall")
        if fp <= 10:
            notes.append("Low FP")
        
        notes_str = ", ".join(notes) if notes else ""
        
        print(f"{threshold: <12.2f} {precision:<12.4f} {recall:<10.4f} {f1:<8.4f} {tp:<6} {fp:<6} {fn: <6} {tn:<6} {notes_str: <20}")
        
        results. append({
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        })
    
    # Find key operating points
    print(f"\n{'='*80}")
    print("🎯 RECOMMENDED OPERATING POINTS")
    print(f"{'='*80}")
    
    # 1. Best F1
    best_f1_result = max(results, key=lambda x:  x['f1'])
    print(f"\n1️⃣  BEST F1 SCORE (Balanced Performance)")
    print(f"   Threshold: {best_f1_result['threshold']:.2f}")
    print(f"   Precision: {best_f1_result['precision']:.4f}")
    print(f"   Recall:     {best_f1_result['recall']:.4f}")
    print(f"   F1:        {best_f1_result['f1']:.4f}")
    print(f"   Detects: {best_f1_result['tp']}/{best_f1_result['tp']+best_f1_result['fn']} accidents")
    print(f"   False alarms: {best_f1_result['fp']}")
    print(f"   💡 Use for:   General purpose detection")
    
    # 2. High precision (closest to 80%)
    high_prec_results = [r for r in results if r['recall'] > 0.20]  # Minimum useful recall
    if high_prec_results:
        best_prec_result = max(high_prec_results, key=lambda x:  x['precision'])
        print(f"\n2️⃣  HIGHEST PRECISION (Minimize False Alarms)")
        print(f"   Threshold: {best_prec_result['threshold']:.2f}")
        print(f"   Precision: {best_prec_result['precision']:.4f}")
        print(f"   Recall:    {best_prec_result['recall']:.4f}")
        print(f"   F1:        {best_prec_result['f1']:.4f}")
        print(f"   Detects: {best_prec_result['tp']}/{best_prec_result['tp']+best_prec_result['fn']} accidents")
        print(f"   False alarms: {best_prec_result['fp']}")
        
        if best_prec_result['precision'] >= 0.80:
            print(f"   ✅ ACHIEVES 80% PRECISION TARGET!")
        else:
            print(f"   ⚠️  Best precision:  {best_prec_result['precision']*100:.1f}% (below 80% target)")
        
        print(f"   💡 Use for:  Insurance reports, dashcam highlights")
    
    # 3. High recall
    best_recall_result = max(results, key=lambda x: x['recall'])
    print(f"\n3️⃣  HIGHEST RECALL (Catch All Accidents)")
    print(f"   Threshold: {best_recall_result['threshold']:.2f}")
    print(f"   Precision: {best_recall_result['precision']:.4f}")
    print(f"   Recall:    {best_recall_result['recall']:.4f}")
    print(f"   F1:        {best_recall_result['f1']:.4f}")
    print(f"   Detects: {best_recall_result['tp']}/{best_recall_result['tp']+best_recall_result['fn']} accidents")
    print(f"   Misses: {best_recall_result['fn']} accidents")
    print(f"   False alarms: {best_recall_result['fp']}")
    print(f"   💡 Use for:  Safety monitoring, early warning")
    
    # 4. Balanced with low false positives
    balanced_results = [r for r in results if r['fp'] <= 50 and r['recall'] >= 0.50]
    if balanced_results: 
        balanced_result = max(balanced_results, key=lambda x: x['f1'])
        print(f"\n4️⃣  BALANCED (Good Performance, Low FP)")
        print(f"   Threshold: {balanced_result['threshold']:.2f}")
        print(f"   Precision: {balanced_result['precision']:.4f}")
        print(f"   Recall:    {balanced_result['recall']:.4f}")
        print(f"   F1:        {balanced_result['f1']:.4f}")
        print(f"   Detects: {balanced_result['tp']}/{balanced_result['tp']+balanced_result['fn']} accidents")
        print(f"   False alarms: {balanced_result['fp']}")
        print(f"   💡 Use for:  Production deployment")
    
    # Summary
    print(f"\n{'='*80}")
    print("📋 SUMMARY & RECOMMENDATIONS")
    print(f"{'='*80}")
    
    print(f"\n✅ Model Performance:")
    print(f"   Best F1: {best_f1_result['f1']:.4f}")
    print(f"   Training time: 16.6 seconds")
    print(f"   Dataset:  1,341 clips (3x augmentation)")
    
    has_80_precision = any(r['precision'] >= 0.80 and r['recall'] >= 0.20 for r in results)
    
    if has_80_precision:
        print(f"\n✅ 80% PRECISION GOAL:   ACHIEVED")
        prec_80_result = [r for r in results if r['precision'] >= 0.80 and r['recall'] >= 0.20][0]
        print(f"   Use threshold: {prec_80_result['threshold']:.2f}")
        print(f"   Precision: {prec_80_result['precision']*100:.1f}%")
        print(f"   Recall:  {prec_80_result['recall']*100:.1f}%")
    else:
        print(f"\n⚠️  80% PRECISION GOAL:  NOT ACHIEVED IN THIS SPLIT")
        print(f"   Maximum precision: {best_prec_result['precision']*100:.1f}%")
        print(f"   This may vary with different training runs")
        print(f"   Consider:")
        print(f"   - Training with different random seed")
        print(f"   - Using K-Fold cross-validation")
        print(f"   - Collecting more data")
    
    print(f"\n💡 Recommended Configuration:")
    print(f"   Model file:   best_model_yolo11n.pth")
    print(f"   For balanced use:   threshold={best_f1_result['threshold']:.2f}")
    print(f"   For low false alarms: threshold={best_prec_result['threshold']:.2f}")
    print(f"   For high recall:   threshold={best_recall_result['threshold']:.2f}")
    
    print(f"\n{'='*80}\n")
    
    return results


if __name__ == '__main__':
    results = analyze_model()