"""
Pre-extract and cache YOLO11n features to disk
Run this ONCE before training - saves hours! 
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from accident_detection_model import TADFrameDataset, YOLOFeatureExtractor
import torchvision.transforms as transforms
import pickle


def cache_all_features():
    print("="*80)
    print("🔥 CACHING YOLO11n FEATURES")
    print("="*80)
    
    # Setup
    print("\n📥 Loading YOLO11n...")
    feature_extractor = YOLOFeatureExtractor('yolo11n.pt', feature_dim=128)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms. ToTensor(),
        transforms. Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    print("\n📂 Loading dataset...")
    dataset = TADFrameDataset(
        root_dir='TAD/frames',
        sequence_length=16,
        transform=transform,
        augment=False
    )
    
    # Create cache directory
    cache_dir = Path('feature_cache')
    cache_dir.mkdir(exist_ok=True)
    
    print(f"\n🚀 Processing {len(dataset)} sequences...")
    print("This will take ~10 minutes.. .\n")
    
    feature_cache = {}
    
    for idx in tqdm(range(len(dataset)), desc="Extracting features"):
        frames_tensor, label = dataset[idx]
        
        # Extract features for this sequence
        sequence_features = []
        for t in range(frames_tensor.shape[0]):
            frame = frames_tensor[t].numpy().transpose(1, 2, 0)
            frame = (frame * 255).astype(np.uint8)
            
            features = feature_extractor.extract_features(frame)
            sequence_features.append(features)
        
        feature_cache[idx] = {
            'features': np.array(sequence_features),
            'label': label. item(),
            'video_dir': str(dataset.sequences[idx])
        }
    
    # Save to disk
    cache_file = cache_dir / 'yolo11n_features_seq16.pkl'
    print(f"\n💾 Saving to {cache_file}...")
    with open(cache_file, 'wb') as f:
        pickle.dump(feature_cache, f)
    
    file_size = cache_file.stat().st_size / 1024 / 1024
    
    print(f"\n{'='*80}")
    print("✅ CACHING COMPLETE!")
    print(f"{'='*80}")
    print(f"📁 Cache file:  {cache_file}")
    print(f"📊 File size: {file_size:.1f} MB")
    print(f"🎯 Total sequences: {len(feature_cache)}")
    print("\n💡 Now run:  python train_fast.py")


if __name__ == '__main__':
    cache_all_features()