"""
Dataset loader for sliding window cached features
"""

import torch
from torch.utils.data import Dataset
import pickle
from pathlib import Path
import numpy as np


class CachedFeatureDatasetSliding(Dataset):
    """Load pre-extracted YOLO features (sliding window version)"""
    
    def __init__(self, cache_file='feature_cache/yolo11n_features_seq16_sliding3.pkl'):
        print(f"Loading cached features from {cache_file}...")
        
        with open(cache_file, 'rb') as f:
            self.cache = pickle.load(f)
        
        self.indices = list(self.cache.keys())
        
        # Get label distribution
        labels = [self.cache[idx]['label'] for idx in self. indices]
        num_accidents = sum(labels)
        num_normal = len(labels) - num_accidents
        
        # Count original videos (by unique video_dir)
        unique_videos = set([self.cache[idx]['video_dir'] for idx in self.indices])
        
        print(f"✅ Loaded {len(self.indices)} cached sequences:")
        print(f"   - {num_accidents} accident clips")
        print(f"   - {num_normal} normal clips")
        print(f"   - From {len(unique_videos)} unique videos")
        print(f"   - Augmentation:  {len(self.indices) / len(unique_videos):.2f}x")
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        data = self.cache[self.indices[idx]]
        features = torch.from_numpy(data['features']).float()
        label = torch.tensor(data['label'], dtype=torch.float32)
        return features, label