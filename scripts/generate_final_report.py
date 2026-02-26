import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, average_precision_score

# --- CONFIGURATION ---
TEST_DIR = "./data/split_maps/test"  # Update this path if your test data is located elsewhere
CHECKPOINT_DIR = "./checkpoints"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'best_resnet_model.pth')
BATCH_SIZE = 32

def generate_report():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading saved model onto {device}...")

    if not os.path.exists(BEST_MODEL_PATH):
        print(f"❌ Error: Could not find model at {BEST_MODEL_PATH}")
        return

    # 1. Rebuild the model skeleton and load weights
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()

    # 2. Load the test data
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder(root=TEST_DIR, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    image_paths = [img[0] for img in test_dataset.imgs]
    accident_idx = test_dataset.class_to_idx.get('accident', 0)

    # 3. Run predictions
    print("Evaluating test images...")
    all_true_labels = []
    all_pred_probs = []
    all_pred_labels = []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            true_y = (y == accident_idx).float().cpu().numpy()
            probs = model(X).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(float)
            
            all_true_labels.extend(true_y)
            all_pred_probs.extend(probs)
            all_pred_labels.extend(preds)

    os.makedirs('reports', exist_ok=True)

    # PHASE 1: GENERATE EXCEL REPORT

    detailed_df = pd.DataFrame({
        'Image_Path': image_paths,
        'True_Label': ['Accident' if label == 1 else 'Normal' for label in all_true_labels],
        'Predicted_Label': ['Accident' if label == 1 else 'Normal' for label in all_pred_labels],
        'Accident_Probability': all_pred_probs
    })
    detailed_df['Correct'] = detailed_df['True_Label'] == detailed_df['Predicted_Label']
    
    report_dict = classification_report(all_true_labels, all_pred_labels, target_names=['Normal', 'Accident'], output_dict=True)
    summary_df = pd.DataFrame(report_dict).transpose()
    
    summary_df.rename(columns={'support': 'Total Images'}, inplace=True)

    excel_path = 'reports/final_evaluation_report.xlsx'
    with pd.ExcelWriter(excel_path) as writer:
        summary_df.to_excel(writer, sheet_name='Summary Metrics')
        detailed_df.to_excel(writer, sheet_name='Detailed Predictions', index=False)
    print(f"Excel report saved to: {excel_path}")

    #STANDARD GRAPHS (CM & ROC)
    cm = confusion_matrix(all_true_labels, all_pred_labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Accident'], yticklabels=['Normal', 'Accident'])
    plt.title('Test Set Confusion Matrix', fontsize=14)
    plt.ylabel('True Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)
    plt.tight_layout()
    plt.savefig('reports/confusion_matrix.png', dpi=300)
    plt.close()

    fpr, tpr, _ = roc_curve(all_true_labels, all_pred_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('reports/roc_curve.png', dpi=300)
    plt.close()

    # THRESHOLD GRAPHS

    thresholds = np.linspace(0.0, 1.0, 100)
    y_true = np.array(all_true_labels)
    p_acc = np.array(all_pred_probs)
    p_norm = 1.0 - p_acc  # Probability of Normal is the inverse
    y_norm = 1.0 - y_true # True Normal labels
    
    prec_acc, rec_acc, f1_acc = [], [], []
    prec_norm, rec_norm, f1_norm = [], [], []
    prec_all, rec_all, f1_all = [], [], []

    for t in thresholds:
        # Accident Metrics
        pred_a = (p_acc >= t).astype(float)
        tp_a = np.sum((pred_a == 1) & (y_true == 1))
        fp_a = np.sum((pred_a == 1) & (y_true == 0))
        fn_a = np.sum((pred_a == 0) & (y_true == 1))
        pa = tp_a / (tp_a + fp_a + 1e-8)
        ra = tp_a / (tp_a + fn_a + 1e-8)
        
        # Normal Metrics
        pred_n = (p_norm >= t).astype(float)
        tp_n = np.sum((pred_n == 1) & (y_norm == 1))
        fp_n = np.sum((pred_n == 1) & (y_norm == 0))
        fn_n = np.sum((pred_n == 0) & (y_norm == 1))
        pn = tp_n / (tp_n + fp_n + 1e-8)
        rn = tp_n / (tp_n + fn_n + 1e-8)
        
        # Append to lists
        prec_acc.append(pa)
        rec_acc.append(ra)
        f1_acc.append(2 * pa * ra / (pa + ra + 1e-8))
        
        prec_norm.append(pn)
        rec_norm.append(rn)
        f1_norm.append(2 * pn * rn / (pn + rn + 1e-8))
        
        # Macro Averages (All Classes)
        prec_all.append((pa + pn) / 2)
        rec_all.append((ra + rn) / 2)
        f1_all.append((f1_acc[-1] + f1_norm[-1]) / 2)

    # Average Precision (AP) Calculations
    ap_acc = average_precision_score(y_true, p_acc)
    ap_norm = average_precision_score(y_norm, p_norm)
    map_all = (ap_acc + ap_norm) / 2

    # Find the optimal threshold based on 'All Classes' F1
    best_idx = np.argmax(f1_all)
    best_thresh = thresholds[best_idx]
    best_f1_val = f1_all[best_idx]

    # --- 1. PR Curve ---
    plt.figure(figsize=(8, 6))
    plt.plot(rec_all, prec_all, color='navy', lw=3, label=f'All Classes mAP@0.5 = {map_all:.3f}')
    plt.plot(rec_acc, prec_acc, color='darkorange', lw=2, label=f'Accident AP = {ap_acc:.3f}')
    plt.plot(rec_norm, prec_norm, color='green', lw=2, label=f'Normal AP = {ap_norm:.3f}')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig('reports/PR_curve.png', dpi=300)
    plt.close()

    # --- 2. F1 Curve ---
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, f1_all, color='navy', lw=3, label='All Classes')
    plt.plot(thresholds, f1_acc, color='darkorange', lw=2, label='Accident')
    plt.plot(thresholds, f1_norm, color='green', lw=2, label='Normal')
    plt.axvline(x=best_thresh, color='red', linestyle='--', label=f'Best Threshold = {best_thresh:.2f}\nMax F1 = {best_f1_val:.2f}')
    plt.xlabel('Confidence Threshold', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.title('F1-Confidence Curve', fontsize=14)
    plt.legend(loc="lower center")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig('reports/F1_curve.png', dpi=300)
    plt.close()

    # --- 3. Precision Curve ---
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, prec_all, color='navy', lw=3, label='All Classes')
    plt.plot(thresholds, prec_acc, color='darkorange', lw=2, label='Accident')
    plt.plot(thresholds, prec_norm, color='green', lw=2, label='Normal')
    plt.xlabel('Confidence Threshold', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Confidence Curve', fontsize=14)
    plt.legend(loc="lower center")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig('reports/P_curve.png', dpi=300)
    plt.close()

    # --- 4. Recall Curve ---
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, rec_all, color='navy', lw=3, label='All Classes')
    plt.plot(thresholds, rec_acc, color='darkorange', lw=2, label='Accident')
    plt.plot(thresholds, rec_norm, color='green', lw=2, label='Normal')
    plt.xlabel('Confidence Threshold', fontsize=12)
    plt.ylabel('Recall', fontsize=12)
    plt.title('Recall-Confidence Curve', fontsize=14)
    plt.legend(loc="lower center")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig('reports/R_curve.png', dpi=300)
    plt.close()

    print("All 6 graphs successfully generated in the 'reports' folder.")

if __name__ == '__main__':
    generate_report()