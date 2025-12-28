"""
Pre-extract YOLO features for faster training
"""

import torch
import numpy as np
import pickle
from tqdm import tqdm
from pathlib import Path
from accident_detection_model import (
    YOLOFeatureExtractor,
    TADFrameDataset,
    TUDATVideoDataset
)
import torchvision.transforms as transforms
import argparse


def extract_tad_features(output_path):
    """Extract features from TAD dataset"""
    print("Extracting features from TAD dataset...")
    
    # Initialize feature extractor
    extractor = YOLOFeatureExtractor('yolov8n.pt')
    
    # Create dataset (no transform needed for feature extraction)
    dataset = TADFrameDataset(
        root_dir='TAD/frames',
        sequence_length=16,
        transform=None
    )
    
    features_list = []
    labels_list = []
    
    for idx in tqdm(range(len(dataset)), desc="Extracting features"):
        video_dir = dataset.sequences[idx]
        label = dataset.labels[idx]
        
        # Get all frames
        frames = sorted(video_dir.glob('*. jpg'))
        sampled_frames = dataset._sample_frames(frames, 16)
        
        # Extract features for each frame
        seq_features = []
        for frame_path in sampled_frames:
            import cv2
            frame = cv2.imread(str(frame_path))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            feat = extractor. extract_features(frame)
            seq_features.append(feat)
        
        features_list.append(np.array(seq_features))
        labels_list.append(label)
    
    # Save to disk
    data = {
        'features': np.array(features_list),
        'labels': np.array(labels_list)
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Features saved to {output_path}")
    print(f"Shape: {data['features'].shape}")


def extract_tudat_features(output_path):
    """Extract features from TU-DAT dataset"""
    print("Extracting features from TU-DAT dataset...")
    
    # Initialize feature extractor
    extractor = YOLOFeatureExtractor('yolov8n.pt')
    
    # Create dataset
    dataset = TUDATVideoDataset(
        video_dir='TU-DAT/TU-DAT/Final_videos',
        annotation_csv='TU-DAT/TU-DAT/annotation.csv',
        sequence_length=16,
        frame_stride=2,
        transform=None
    )
    
    features_list = []
    labels_list = []
    
    for idx in tqdm(range(len(dataset)), desc="Extracting features"):
        video_path = dataset.videos[idx]
        label = dataset.labels[idx]
        collision_time = dataset.collision_times[idx]
        
        # Extract frames
        frames = dataset._extract_frames(video_path, collision_time)
        
        # Extract features for each frame
        seq_features = []
        for frame in frames:
            feat = extractor.extract_features(frame)
            seq_features. append(feat)
        
        features_list.append(np. array(seq_features))
        labels_list.append(label)
    
    # Save to disk
    data = {
        'features': np.array(features_list),
        'labels': np.array(labels_list)
    }
    
    with open(output_path, 'wb') as f:
        pickle. dump(data, f)
    
    print(f"Features saved to {output_path}")
    print(f"Shape: {data['features'].shape}")


if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='Extract YOLO features')
    parser.add_argument('--dataset', type=str, choices=['tad', 'tudat'],
                       required=True, help='Which dataset to process')
    parser.add_argument('--output', type=str, required=True,
                       help='Output pickle file')
    
    args = parser.parse_args()
    
    if args.dataset == 'tad':
        extract_tad_features(args.output)
    elif args.dataset == 'tudat':
        extract_tudat_features(args. output)