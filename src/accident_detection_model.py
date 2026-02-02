"""
Traffic Accident Detection using YOLO + LSTM/Transformer
Combines spatial features from YOLO with temporal modeling
IMPROVED VERSION with better feature extraction and handling
"""

import torch
import torch. nn as nn
from ultralytics import YOLO
import numpy as np
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import pandas as pd
from pathlib import Path
import torchvision.transforms as transforms


# ============================================================================
# 1. YOLO Feature Extractor
# ============================================================================

class YOLOFeatureExtractor:      
    """Extract spatial features from frames using YOLO"""
    
    def __init__(self, model_name='yolov8n. pt', feature_dim=128):
        """
        Args:  
            model_name:  YOLO model variant (yolov8n, yolov8s, yolov8m, etc.)
            feature_dim: Output feature dimension
        """
        self. model = YOLO(model_name)
        self.model.eval()
        self.feature_dim = feature_dim
        
    def extract_features(self, frame):
        """
        Extract features from a single frame
        
        Args:
            frame: numpy array (H, W, C)
            
        Returns:
            feature_vector: numpy array of shape (feature_dim,)
        """
        results = self.model(frame, verbose=False)
        
        # Extract bounding boxes, classes, and confidences
        boxes = results[0].boxes
        
        if len(boxes) > 0:
            # Get detection features
            xyxy = boxes.xyxy.cpu().numpy()  # Bounding boxes
            conf = boxes.conf.cpu().numpy()  # Confidence scores
            cls = boxes.cls.cpu().numpy()    # Class IDs
            
            # Create feature vector
            features = self._create_feature_vector(xyxy, conf, cls, frame.shape)
        else:
            # No detections - return zero vector
            features = np.zeros(self.feature_dim)
            
        return features
    
    def _create_feature_vector(self, xyxy, conf, cls, frame_shape):
        """Create a fixed-size feature vector from detections"""
        h, w = frame_shape[: 2]
        
        # Basic statistics
        num_objects = len(xyxy)
        avg_conf = np.mean(conf)
        max_conf = np.max(conf)
        min_conf = np.min(conf)
        std_conf = np.std(conf)
        
        # Class distribution (80 COCO classes)
        class_dist = np.bincount(cls. astype(int), minlength=80) / max(num_objects, 1)
        
        # Spatial features
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        normalized_areas = areas / (h * w)
        
        avg_area = np.mean(normalized_areas) if len(areas) > 0 else 0
        max_area = np.max(normalized_areas) if len(areas) > 0 else 0
        std_area = np.std(normalized_areas) if len(areas) > 0 else 0
        
        # Centroid positions (normalized)
        centroids_x = (xyxy[:, 0] + xyxy[: , 2]) / (2 * w)
        centroids_y = (xyxy[:, 1] + xyxy[:, 3]) / (2 * h)
        
        mean_x = np.mean(centroids_x) if len(centroids_x) > 0 else 0.5
        mean_y = np. mean(centroids_y) if len(centroids_y) > 0 else 0.5
        std_x = np.std(centroids_x) if len(centroids_x) > 0 else 0
        std_y = np.std(centroids_y) if len(centroids_y) > 0 else 0
        
        # Aspect ratios
        widths = xyxy[:, 2] - xyxy[:, 0]
        heights = xyxy[: , 3] - xyxy[:, 1]
        aspect_ratios = widths / (heights + 1e-6)
        avg_aspect = np.mean(aspect_ratios) if len(aspect_ratios) > 0 else 1.0
        
        # Edge proximity (how close objects are to frame edges)
        left_prox = np.mean(xyxy[:, 0] / w) if len(xyxy) > 0 else 0
        right_prox = np. mean((w - xyxy[:, 2]) / w) if len(xyxy) > 0 else 0
        top_prox = np.mean(xyxy[:, 1] / h) if len(xyxy) > 0 else 0
        bottom_prox = np.mean((h - xyxy[:, 3]) / h) if len(xyxy) > 0 else 0
        
        # Count of important classes (vehicles, people)
        vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        person_class = [0]  # person
        
        num_vehicles = np.sum([np.sum(cls == c) for c in vehicle_classes])
        num_persons = np.sum([np.sum(cls == c) for c in person_class])
        
        # Density features
        density = num_objects / (h * w) * 1000000  # objects per megapixel
        
        # Combine all features
        basic_features = np.array([
            num_objects,
            avg_conf, max_conf, min_conf, std_conf,
            avg_area, max_area, std_area,
            mean_x, mean_y, std_x, std_y,
            avg_aspect,
            left_prox, right_prox, top_prox, bottom_prox,
            num_vehicles, num_persons,
            density
        ])
        
        # Take top 20 most common classes for class distribution
        top_classes = class_dist[: 20]
        
        # Combine everything
        feature_vector = np.concatenate([
            basic_features,  # 20 features
            top_classes      # 20 features
        ])  # Total:  40 features
        
        # Pad or truncate to desired feature_dim
        if len(feature_vector) < self.feature_dim:
            feature_vector = np.pad(feature_vector, 
                                   (0, self.feature_dim - len(feature_vector)), 
                                   mode='constant')
        else:
            feature_vector = feature_vector[:self.feature_dim]
            
        return feature_vector


# ============================================================================
# 2. Datasets with Augmentation
# ============================================================================

class TADFrameDataset(Dataset):
    """Dataset for pre-extracted frames (TAD dataset)"""
    
    def __init__(self, root_dir, sequence_length=16, transform=None, augment=False):
        """
        Args:
            root_dir: Path to TAD/frames/
            sequence_length: Number of frames per sequence
            transform:  Optional image transformations
            augment: Whether to apply temporal augmentation
        """
        self.root_dir = Path(root_dir)
        self.sequence_length = sequence_length
        self.transform = transform
        self.augment = augment
        
        # Collect all video directories
        self.sequences = []
        self.labels = []
        
        print(f"Scanning dataset: {self.root_dir}")
        
        # Process abnormal (accident) videos
        abnormal_dir = self.root_dir / 'abnormal'
        abnormal_count = 0
        if abnormal_dir.exists():
            for item in abnormal_dir.iterdir():
                if item.is_dir():
                    self.sequences.append(item)
                    self.labels.append(1)  # Accident
                    abnormal_count += 1
        
        # Process normal videos
        normal_dir = self.root_dir / 'normal'
        normal_count = 0
        if normal_dir.exists():
            for item in normal_dir.iterdir():
                if item.is_dir():
                    self.sequences.append(item)
                    self. labels.append(0)  # No accident
                    normal_count += 1
        
        print(f"✓ Found {len(self.sequences)} sequences:")
        print(f"  - {abnormal_count} accident videos")
        print(f"  - {normal_count} normal videos")
        
        # Calculate class weights for imbalanced dataset
        if abnormal_count > 0 and normal_count > 0:
            self.pos_weight = normal_count / abnormal_count
            print(f"  - Class imbalance ratio: {self.pos_weight:.2f}")
        else:
            self.pos_weight = 1.0
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        video_dir = self.sequences[idx]
        label = self.labels[idx]
        
        # Get all frames in directory - support multiple extensions
        image_extensions = ['*. jpg', '*.jpeg', '*. png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
        frames = []
        for ext in image_extensions:
            frames.extend(video_dir.glob(ext))
        frames = sorted(frames)
        
        # Safety check
        if len(frames) == 0:
            print(f"WARNING: No frames in {video_dir}!  Creating dummy frames.")
            dummy_frame = np.zeros((224, 224, 3), dtype=np.uint8)
            if self.transform:
                dummy_frame = self.transform(dummy_frame)
            return torch.stack([dummy_frame] * self.sequence_length), torch.tensor(label, dtype=torch.float32)
        
        # Sample frames uniformly (with optional augmentation)
        sampled_frames = self._sample_frames(frames, self.sequence_length)
        
        # Load and process frames
        frame_list = []
        for frame_path in sampled_frames:
            img = cv2.imread(str(frame_path))
            if img is None:
                print(f"Warning: Failed to load {frame_path}")
                img = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                img = self.transform(img)
            
            frame_list.append(img)
        
        return torch.stack(frame_list), torch.tensor(label, dtype=torch.float32)
    
    def _sample_frames(self, frames, num_samples):
        """Uniformly sample frames from sequence with optional augmentation"""
        if len(frames) == 0:
            raise ValueError("Cannot sample from empty frame list")
        
        # Temporal augmentation (random frame stride during training)
        if self.augment and np.random.rand() > 0.5:
            # Random temporal stride
            stride = np.random.choice([1, 2, 3])
            start_idx = np.random.randint(0, max(1, len(frames) - num_samples * stride))
            indices = [min(start_idx + i * stride, len(frames) - 1) for i in range(num_samples)]
            sampled = [frames[i] for i in indices]
        elif len(frames) <= num_samples:
            # Repeat last frame if not enough frames
            sampled = list(frames) + [frames[-1]] * (num_samples - len(frames))
        else:
            # Uniform sampling
            indices = np.linspace(0, len(frames) - 1, num_samples, dtype=int)
            sampled = [frames[i] for i in indices]
        
        return sampled


class TUDATVideoDataset(Dataset):
    """Dataset for raw videos with annotations (TU-DAT dataset)"""
    
    def __init__(self, video_dir, annotation_csv, sequence_length=16, 
                 frame_stride=1, transform=None):
        """
        Args: 
            video_dir: Path to TU-DAT/Final_videos/
            annotation_csv: Path to annotation. csv
            sequence_length: Number of frames per sequence
            frame_stride: Sample every Nth frame
            transform: Optional image transformations
        """
        self.video_dir = Path(video_dir)
        self.sequence_length = sequence_length
        self.frame_stride = frame_stride
        self.transform = transform
        
        # Load annotations
        self.annotations = pd.read_csv(annotation_csv)
        
        # Collect videos
        self.videos = []
        self.labels = []
        self.collision_times = []
        
        # Process positive videos (with accidents)
        positive_dir = self.video_dir / 'Positive_VIdeos'
        if positive_dir.exists():
            for video_path in positive_dir.glob('*.mov'):
                video_name = video_path.name
                collision_row = self.annotations[self.annotations['video'] == video_name]
                if not collision_row.empty:
                    collision_time = collision_row. iloc[0]['pointofCollision']
                    self.videos.append(video_path)
                    self.labels.append(1)
                    self.collision_times. append(collision_time)
        
        # Process negative videos (without accidents)
        negative_dir = self.video_dir / 'Negative_VIdeos'
        if negative_dir.exists():
            for video_path in negative_dir.glob('*.mov'):
                self.videos.append(video_path)
                self.labels.append(0)
                self.collision_times.append(None)
        
        print(f"Found {len(self.videos)} videos:")
        print(f"  - {sum(self.labels)} with accidents")
        print(f"  - {len(self.labels) - sum(self.labels)} without accidents")
    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        video_path = self.videos[idx]
        label = self.labels[idx]
        collision_time = self.collision_times[idx]
        
        frames = self._extract_frames(video_path, collision_time)
        
        frame_list = []
        for frame in frames:
            if self.transform:
                frame = self.transform(frame)
            frame_list.append(frame)
        
        return torch.stack(frame_list), torch.tensor(label, dtype=torch.float32)
    
    def _extract_frames(self, video_path, collision_time=None):
        """Extract frames from video, focusing around collision time if available"""
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if collision_time is not None: 
            collision_frame = int(collision_time * fps)
            start_frame = max(0, collision_frame - self.sequence_length * self.frame_stride // 2)
        else:
            max_start = max(0, total_frames - self.sequence_length * self.frame_stride)
            start_frame = np.random.randint(0, max_start + 1) if max_start > 0 else 0
        
        frames = []
        frame_idx = 0
        
        while len(frames) < self.sequence_length:
            ret, frame = cap.read()
            if not ret:
                if frames:
                    frames.append(frames[-1]. copy())
                else:
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
                continue
            
            if frame_idx >= start_frame and (frame_idx - start_frame) % self.frame_stride == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            frame_idx += 1
        
        cap.release()
        
        while len(frames) < self.sequence_length:
            frames.append(frames[-1].copy() if frames else np.zeros((224, 224, 3), dtype=np.uint8))
        
        return frames[: self.sequence_length]


# ============================================================================
# 3. Improved Temporal Models
# ============================================================================

class AccidentDetectionLSTM(nn.Module):
    """LSTM-based temporal model with attention"""
    
    def __init__(self, input_dim=128, hidden_dim=256, num_layers=2, dropout=0.4):
        super(AccidentDetectionLSTM, self).__init__()
        
        self. lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True  # Bidirectional for better context
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)  # *2 for bidirectional
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Classification head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        lstm_out = self.layer_norm(lstm_out)
        
        # Attention mechanism
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq_len, 1)
        attended = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, hidden*2)
        
        # Classification
        output = self.fc(self.dropout(attended))
        
        return output


class AccidentDetectionTransformer(nn.Module):
    """Transformer-based temporal model"""
    
    def __init__(self, input_dim=128, d_model=256, nhead=8, 
                 num_layers=4, dropout=0.3):
        super(AccidentDetectionTransformer, self).__init__()
        
        self. input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'  # GELU activation
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Classification head with more capacity
        self.fc = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        
        transformer_out = self.transformer(x)
        transformer_out = self.layer_norm(transformer_out)
        
        # Use both mean and max pooling
        mean_pool = torch.mean(transformer_out, dim=1)
        max_pool, _ = torch.max(transformer_out, dim=1)
        pooled = mean_pool + max_pool  # Combine both
        
        output = self.fc(pooled)
        
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x. size(1), :]
        return self.dropout(x)


# ============================================================================
# 4. Improved Pipeline
# ============================================================================

class AccidentDetectionPipeline: 
    """Complete pipeline for accident detection"""
    
    def __init__(self, yolo_model='yolov8n.pt', temporal_model='lstm', 
                 input_dim=128, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.input_dim = input_dim
        
        # Initialize YOLO feature extractor
        self.feature_extractor = YOLOFeatureExtractor(yolo_model, feature_dim=input_dim)
        
        # Initialize temporal model
        if temporal_model == 'lstm': 
            self.model = AccidentDetectionLSTM(input_dim=input_dim)
        elif temporal_model == 'transformer':
            self.model = AccidentDetectionTransformer(input_dim=input_dim)
        else:
            raise ValueError(f"Unknown temporal model:  {temporal_model}")
        
        self.model = self.model.to(self. device)
        
        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p. numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nModel:  {temporal_model. upper()}")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Device: {self.device}")
    
    def extract_features_from_frames(self, frames):
        """Extract features from a sequence of frames"""
        if isinstance(frames, torch.Tensor):
            frames = frames.cpu().numpy()
            if frames.ndim == 5:  # (B, T, C, H, W)
                batch_size, seq_len = frames.shape[:2]
                features_list = []
                
                for b in range(batch_size):
                    seq_features = []
                    for t in range(seq_len):
                        frame = frames[b, t]. transpose(1, 2, 0)  # (H, W, C)
                        frame = (frame * 255).astype(np.uint8)
                        feat = self.feature_extractor. extract_features(frame)
                        seq_features.append(feat)
                    features_list.append(np.array(seq_features))
                
                features = torch.from_numpy(np.array(features_list)).float()
            else: 
                raise ValueError("Expected 5D tensor (B, T, C, H, W)")
        else:
            features = []
            for frame in frames:
                feat = self.feature_extractor. extract_features(frame)
                features.append(feat)
            features = torch.from_numpy(np.array(features)).float().unsqueeze(0)
        
        return features. to(self.device)
    
    def predict(self, frames):
        """Predict accident probability from frames"""
        self.model.eval()
        with torch.no_grad():
            features = self.extract_features_from_frames(frames)
            output = self.model(features)
            return output.cpu().item()