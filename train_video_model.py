import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
import numpy as np
import torchvision.transforms as transforms
from src.video_model import VideoAccidentDetector
from src.video_dataset import TemporalAccidentDataset

def train():
    BATCH_SIZE = 8
    EPOCHS = 50
    LR = 0.0001
    SEQ_LENGTH = 16
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Annotations
    csv_path = 'annotations/accidents_cleaned.csv' # Adjust to your actual CSV filename
    frames_dir = 'data/cadp/extracted_frames'              # Directory containing your video folders
    
    df = pd.read_csv(csv_path)
    
    # Ensure column names match expected formats
    # Assuming columns are: videoid, startframe, endframe
    
    # 2. Strict Video-Level Split
    unique_videos = df['videoid'].unique()
    train_vids, val_vids = train_test_split(unique_videos, test_size=0.2, random_state=42)
    
    # Filter the dataframes
    train_df = df[df['videoid'].isin(train_vids)]
    val_df = df[df['videoid'].isin(val_vids)]
    
    print(f"Training on {len(train_vids)} videos, Validating on {len(val_vids)} videos")

    # 3. Transforms
    transform_pipeline = transforms.Compose([
        transforms.Resize((112, 112), antialias=True),
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
    ])

    # 4. Datasets
    train_dataset = TemporalAccidentDataset(train_df, frames_dir, sequence_length=SEQ_LENGTH, transform=transform_pipeline, augment=True)
    val_dataset = TemporalAccidentDataset(val_df, frames_dir, sequence_length=SEQ_LENGTH, transform=transform_pipeline, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 5. Model Setup
    model = VideoAccidentDetector().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # 6. Training Loop (Standard PyTorch loop)
    best_f1 = 0.0
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        for videos, labels in train_loader:
            videos, labels = videos.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for videos, labels in val_loader:
                videos = videos.to(DEVICE)
                outputs = model(videos)
                preds = (outputs > 0.5).float().cpu().numpy()
                val_preds.extend(preds)
                val_targets.extend(labels.numpy())
                
        f1 = f1_score(val_targets, val_preds, zero_division=0)
        print(f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.4f} | Val F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), 'best_temporal_model.pth')

if __name__ == "__main__":
    train()