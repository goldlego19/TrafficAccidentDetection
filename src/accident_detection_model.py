import torch
import torch.nn as nn
from ultralytics import YOLO
import numpy as np

class YOLOFeatureExtractor:
    def __init__(self, model_path='yolo11n.pt', feature_dim=128):
        self.model = YOLO(model_path)
        self.feature_dim = feature_dim

    def extract_features(self, img):
        # 1. INFERENCE SETTINGS
        # conf=0.15: Lower threshold to catch faint cars in bad video
        # classes=[2,3,5,7]: ONLY detect Car(2), Motorcycle(3), Bus(5), Truck(7). Ignore Boats(8)!
        # verbose=False: Keep terminal clean
        results = self.model(img, conf=0.15, iou=0.5, classes=[2,3,5,7], verbose=False)
        
        result = results[0]
        boxes = result.boxes.data.cpu().numpy() # [x1, y1, x2, y2, conf, cls]
        
        if len(boxes) == 0:
            return torch.zeros(self.feature_dim).numpy()
            
        # 2. NORMALIZE COORDINATES (0-1)
        h, w, _ = img.shape
        boxes[:, 0] /= w 
        boxes[:, 1] /= h 
        boxes[:, 2] /= w 
        boxes[:, 3] /= h 
        
        # 3. SPATIAL SORTING (Left-to-Right)
        # Crucial for video stability
        boxes = boxes[boxes[:, 0].argsort()]
        
        # Take top 5 objects
        boxes = boxes[:5]
        
        # Flatten and pad
        features = boxes.flatten()
        if len(features) < self.feature_dim:
            features = torch.tensor(features)
            features = torch.nn.functional.pad(features, (0, self.feature_dim - len(features)))
        else:
            features = torch.tensor(features[:self.feature_dim])
            
        return features.numpy()

class AccidentDetectionLSTM(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=64, num_layers=1, dropout=0.5):
        super(AccidentDetectionLSTM, self).__init__()
        
        self.input_norm = nn.LayerNorm(input_dim)
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.input_norm(x)
        lstm_out, _ = self.lstm(x)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        attended = torch.sum(attention_weights * lstm_out, dim=1)
        output = self.fc(self.dropout(attended))
        return output