"""
Cache YOLO features with SLIDING WINDOW
Creates 3 clips per video for data augmentation
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from accident_detection_model import YOLOFeatureExtractor
import torchvision.transforms as transforms
import pickle
import cv2


def extract_clip_features(frame_paths, feature_extractor, transform, start_idx, end_idx, sequence_length=16):
    """Extract features for one clip"""
    # Get frames in this clip range
    clip_frames = frame_paths[start_idx:end_idx]
    
    if len(clip_frames) == 0:
        return None
    
    # Sample uniformly within clip
    if len(clip_frames) <= sequence_length:
        sampled_frames = list(clip_frames) + [clip_frames[-1]] * (sequence_length - len(clip_frames))
    else:
        indices = np.linspace(0, len(clip_frames) - 1, sequence_length, dtype=int)
        sampled_frames = [clip_frames[i] for i in indices]
    
    # Extract YOLO features
    features_list = []
    for frame_path in sampled_frames: 
        img = cv2.imread(str(frame_path))
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Transform
        if transform: 
            img = transform(img)
            img_np = img.numpy().transpose(1, 2, 0)
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img
        
        # YOLO features
        feat = feature_extractor. extract_features(img_np)
        features_list.append(feat)
    
    return np.array(features_list)


def cache_sliding_window_features():
    print("="*80)
    print("🔥 CACHING YOLO11n FEATURES WITH SLIDING WINDOW")
    print("="*80)
    
    # Config
    SEQUENCE_LENGTH = 16
    NUM_CLIPS = 3  # 3 clips per video
    
    print(f"\n⚙️  Configuration:")
    print(f"   Sequence length: {SEQUENCE_LENGTH}")
    print(f"   Clips per video: {NUM_CLIPS}")
    print(f"   Expected samples:  ~{447 * NUM_CLIPS} (3x original)")
    
    # Setup
    print("\n📥 Loading YOLO11n...")
    feature_extractor = YOLOFeatureExtractor('yolo11n.pt', feature_dim=128)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    print("\n📂 Loading TAD dataset...")
    dataset_root = Path('TAD/frames')
    
    # Collect all video directories
    sequences = []
    labels = []
    
    # Abnormal
    abnormal_dir = dataset_root / 'abnormal'
    if abnormal_dir.exists():
        for item in abnormal_dir.iterdir():
            if item.is_dir():
                sequences.append(item)
                labels.append(1)
    
    # Normal
    normal_dir = dataset_root / 'normal'
    if normal_dir. exists():
        for item in normal_dir.iterdir():
            if item.is_dir():
                sequences.append(item)
                labels.append(0)
    
    print(f"✓ Found {len(sequences)} videos")
    print(f"  - {sum(labels)} accident videos")
    print(f"  - {len(labels) - sum(labels)} normal videos")
    
    # Create cache directory
    cache_dir = Path('feature_cache')
    cache_dir.mkdir(exist_ok=True)
    
    print(f"\n🚀 Processing with sliding window (3 clips per video)...")
    print("This will take ~20-30 minutes.. .\n")
    
    feature_cache = {}
    cache_idx = 0
    
    for video_idx, (video_dir, label) in enumerate(tqdm(zip(sequences, labels), total=len(sequences), desc="Processing videos")):
        # Get all frames
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']
        frames = []
        for ext in image_extensions:
            frames.extend(video_dir.glob(ext))
        frames = sorted(frames)
        
        if len(frames) == 0:
            print(f"WARNING: No frames in {video_dir}")
            continue
        
        total_frames = len(frames)
        
        # Create 3 clips
        clip_size = total_frames // NUM_CLIPS
        
        for clip_num in range(NUM_CLIPS):
            start_idx = clip_num * clip_size
            
            if clip_num == NUM_CLIPS - 1:
                # Last clip:  take remaining frames
                end_idx = total_frames
            else: 
                end_idx = start_idx + clip_size
            
            # Extract features for this clip
            features = extract_clip_features(
                frames, feature_extractor, transform,
                start_idx, end_idx, SEQUENCE_LENGTH
            )
            
            if features is not None:
                feature_cache[cache_idx] = {
                    'features': features,
                    'label': label,
                    'video_dir': str(video_dir),
                    'clip_num': clip_num + 1,
                    'clip_range': f"frames {start_idx}-{end_idx}"
                }
                cache_idx += 1
    
    # Save to disk
    cache_file = cache_dir / f'yolo11n_features_seq{SEQUENCE_LENGTH}_sliding{NUM_CLIPS}.pkl'
    print(f"\n💾 Saving to {cache_file}...")
    with open(cache_file, 'wb') as f:
        pickle.dump(feature_cache, f)
    
    file_size = cache_file.stat().st_size / 1024 / 1024
    
    print(f"\n{'='*80}")
    print("✅ CACHING COMPLETE!")
    print(f"{'='*80}")
    print(f"📁 Cache file:  {cache_file}")
    print(f"📊 File size: {file_size:.1f} MB")
    print(f"🎯 Total samples: {len(feature_cache)}")
    print(f"   - Original videos: {len(sequences)}")
    print(f"   - Clips per video: {NUM_CLIPS}")
    print(f"   - Augmentation factor: {len(feature_cache) / len(sequences):.2f}x")
    
    # Class distribution
    num_accidents = sum([1 for v in feature_cache.values() if v['label'] == 1])
    num_normal = len(feature_cache) - num_accidents
    print(f"\n📈 Sample distribution:")
    print(f"   - {num_accidents} accident clips")
    print(f"   - {num_normal} normal clips")
    print(f"   - Ratio: {num_normal / num_accidents:.2f}: 1")
    
    print("\n💡 Now run:  python train_fast_sliding. py")


if __name__ == '__main__':
    cache_sliding_window_features()