import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import logging
from tqdm import tqdm 

# PHASE 1: CONFIGURATION & SETUP

# Point directly to our newly split folders
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'split_mapsNew')
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VAL_DIR = os.path.join(BASE_DIR, 'val')

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BATCH_SIZE = 32 # Number of images the model looks at before updating its weights
EPOCHS = 20     # Number of times the model will see the entire dataset

logging.basicConfig(level=logging.INFO, format='%(message)s')

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"STARTING RESNET18 Training ON {device}")

    # PHASE 2: DATA PREPARATION & LOADING

    # Standardise all images so the network can process them.
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), # Converts image pixels to numbers between 0 and 1
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Because the images are physically separated, we just point ImageFolder to the correct directory
    train_data = datasets.ImageFolder(root=TRAIN_DIR, transform=train_transform)
    val_data = datasets.ImageFolder(root=VAL_DIR, transform=val_transform)
    
    # Explicitly find which internal number PyTorch assigned to the 'accident' folder
    accident_idx = train_data.class_to_idx.get('accident', 0)
    
    # DataLoaders package our data into batches and feed them to the model
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    # PHASE 3: MODEL ARCHITECTURE (Transfer Learning)

    # Load a pre-trained ResNet18 model 
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # 'Freeze' the pre-trained weights so we do not overwrite its fundamental knowledge
    for param in model.parameters():
        param.requires_grad = False

    # Replace the model's final classification layer with our own custom neural network head
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256), # Compress the features down to 256 nodes
        nn.ReLU(),                            # Activation function to handle non-linear patterns
        nn.Dropout(0.5),                      # Randomly turn off 50% of nodes to prevent memorisation 
        nn.Linear(256, 1),                    # Output a single number
        nn.Sigmoid()                          # Squash that number into a probability between 0 and 1
    )
    model = model.to(device)

    # PHASE 4: THE OPTIMISER & LOSS FUNCTION

    # The optimiser acts as the steering wheel, updating our custom layer's weights based on errors.
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001, weight_decay=1e-4)
    # Binary Cross Entropy Loss measures how far our predictions are from the true labels
    criterion = nn.BCELoss()
    
    best_f1 = 0.0

    # PHASE 5: THE TRAINING LOOP

    for epoch in range(EPOCHS):
        model.train() # Set model to training mode
        total_loss = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [Train]", leave=False)
        
        for X, y in train_pbar:
            X, y = X.to(device), (y == accident_idx).float().unsqueeze(1).to(device)
            
            optimizer.zero_grad()         # A. Clear out old gradients from the last step
            loss = criterion(model(X), y) # B. Make a prediction and calculate the error
            loss.backward()               # C. Work backwards to calculate fixes (gradients)
            optimizer.step()              # D. Apply the fixes to the model weights
            
            total_loss += loss.item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}") 
            
        # PHASE 6: THE VALIDATION LOOP & METRICS

        model.eval() # Set model to evaluation mode
        tp, fp, tn, fn = 0, 0, 0, 0 
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [Val]  ", leave=False)
        
        with torch.no_grad(): # Disable gradient calculation to save memory 
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

        # Save the model if it's the best one yet
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'best_resnet_model2.pth'))
            logging.info(f"New Best Model Saved! (F1: {f1:.3f})")

    # PHASE 7: YOLO-STYLE FINAL SUMMARY

    logging.info(f"\n{EPOCHS} epochs completed. Generating final report...")
    best_model_path = os.path.join(CHECKPOINT_DIR, 'best_resnet_model2.pth')
    
    # Load the absolute best weights back into the model for the final test
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    tp, fp, tn, fn = 0, 0, 0, 0
    total_images = 0
    accident_targets, normal_targets = 0, 0
    
    with torch.no_grad():
        for X, y in val_loader: # Running the summary on the validation set
            X, y = X.to(device), (y == accident_idx).float().unsqueeze(1).to(device)
            preds = (model(X) > 0.5).float()
            
            tp += ((preds == 1) & (y == 1)).sum().item()
            fp += ((preds == 1) & (y == 0)).sum().item()
            tn += ((preds == 0) & (y == 0)).sum().item()
            fn += ((preds == 0) & (y == 1)).sum().item()
            
            total_images += y.size(0)
            accident_targets += (y == 1).sum().item()
            normal_targets += (y == 0).sum().item()

    # Calculate final YOLO-style metrics
    p_acc = tp / (tp + fp + 1e-8)
    r_acc = tp / (tp + fn + 1e-8)
    f1_acc = 2 * (p_acc * r_acc) / (p_acc + r_acc + 1e-8)
    
    p_norm = tn / (tn + fn + 1e-8)
    r_norm = tn / (tn + fp + 1e-8)
    f1_norm = 2 * (p_norm * r_norm) / (p_norm + r_norm + 1e-8)
    
    p_macro = (p_acc + p_norm) / 2
    r_macro = (r_acc + r_norm) / 2
    f1_macro = (f1_acc + f1_norm) / 2
    acc_all = (tp + tn) / total_images

    logging.info(f"\n{'Class':<12} {'Images':>8} {'Targets':>8} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Accuracy':>10}")
    logging.info(f"{'all':<12} {total_images:>8} {total_images:>8} {p_macro:>10.3f} {r_macro:>10.3f} {f1_macro:>10.3f} {acc_all:>10.3f}")
    logging.info(f"{'accident':<12} {total_images:>8} {int(accident_targets):>8} {p_acc:>10.3f} {r_acc:>10.3f} {f1_acc:>10.3f} {'-':>10}")
    logging.info(f"{'normal':<12} {total_images:>8} {int(normal_targets):>8} {p_norm:>10.3f} {r_norm:>10.3f} {f1_norm:>10.3f} {'-':>10}")
    logging.info(f"\nModel weights saved to: {CHECKPOINT_DIR}")

if __name__ == '__main__':
    train()