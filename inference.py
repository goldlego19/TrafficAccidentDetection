"""
Inference script for traffic accident detection
"""

import cv2
import numpy as np
import torch
from accident_detection_model import AccidentDetectionPipeline
import argparse
from pathlib import Path
import torchvision.transforms as transforms


def get_transforms():
    """Define image transformations"""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms. Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def process_video(video_path, pipeline, sequence_length=16, stride=8):
    """
    Process a video and detect accidents with sliding window
    
    Args:
        video_path: Path to video file
        pipeline: AccidentDetectionPipeline instance
        sequence_length: Number of frames per prediction
        stride: How many frames to slide the window
        
    Returns:
        predictions: List of (timestamp, probability) tuples
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    transform = get_transforms()
    predictions = []
    frame_buffer = []
    frame_idx = 0
    
    print(f"Processing video: {video_path}")
    print(f"Total frames: {total_frames}, FPS: {fps}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Transform frame
        frame_tensor = transform(frame_rgb)
        frame_buffer.append(frame_tensor)
        
        # Process when we have enough frames
        if len(frame_buffer) == sequence_length:
            # Stack frames
            frame_sequence = torch.stack(frame_buffer).unsqueeze(0)  # (1, T, C, H, W)
            
            # Predict
            probability = pipeline.predict(frame_sequence)
            timestamp = frame_idx / fps
            
            predictions.append((timestamp, probability))
            print(f"Frame {frame_idx}/{total_frames} | "
                  f"Time: {timestamp:.2f}s | "
                  f"Accident Probability: {probability:.4f}")
            
            # Slide window
            frame_buffer = frame_buffer[stride:]
        
        frame_idx += 1
    
    cap.release()
    
    return predictions


def process_video_realtime(video_path, pipeline, threshold=0.7):
    """
    Process video with real-time visualization
    
    Args:
        video_path: Path to video file
        pipeline: AccidentDetectionPipeline instance
        threshold:  Probability threshold for accident detection
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    transform = get_transforms()
    frame_buffer = []
    sequence_length = 16
    
    print("Processing video in real-time mode...")
    print("Press 'q' to quit")
    
    while True: 
        ret, frame = cap.read()
        if not ret: 
            break
        
        # Convert and transform
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = transform(frame_rgb)
        frame_buffer.append(frame_tensor)
        
        # Keep only last sequence_length frames
        if len(frame_buffer) > sequence_length:
            frame_buffer. pop(0)
        
        # Predict if we have enough frames
        if len(frame_buffer) == sequence_length:
            frame_sequence = torch.stack(frame_buffer).unsqueeze(0)
            probability = pipeline.predict(frame_sequence)
            
            # Visualize
            color = (0, 0, 255) if probability > threshold else (0, 255, 0)
            label = f"ACCIDENT:  {probability:.2f}" if probability > threshold else f"Normal:  {probability:.2f}"
            
            cv2.putText(frame, label, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Draw probability bar
            bar_width = int(probability * 300)
            cv2.rectangle(frame, (10, 50), (310, 80), (255, 255, 255), 2)
            cv2.rectangle(frame, (10, 50), (10 + bar_width, 80), color, -1)
        
        # Display
        cv2.imshow('Accident Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Accident Detection Inference')
    parser.add_argument('--video', type=str, required=True, 
                       help='Path to input video')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model weights')
    parser.add_argument('--temporal-model', type=str, choices=['lstm', 'transformer'],
                       default='lstm', help='Type of temporal model')
    parser.add_argument('--mode', type=str, choices=['sliding', 'realtime'],
                       default='sliding', help='Processing mode')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='Detection threshold')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file for predictions (CSV)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    print("Initializing pipeline...")
    pipeline = AccidentDetectionPipeline(
        yolo_model='yolov8n.pt',
        temporal_model=args. temporal_model,
        input_dim=48,
        device='cuda'
    )
    
    # Load trained weights
    print(f"Loading model from {args.model}...")
    pipeline.model.load_state_dict(torch.load(args.model))
    pipeline.model.eval()
    
    # Process video
    if args.mode == 'sliding': 
        predictions = process_video(args.video, pipeline)
        
        # Find accident moments
        accidents = [(t, p) for t, p in predictions if p > args.threshold]
        
        if accidents:
            print(f"\n{'='*50}")
            print(f"ACCIDENTS DETECTED:  {len(accidents)}")
            print(f"{'='*50}")
            for timestamp, prob in accidents:
                print(f"  Time: {timestamp:.2f}s | Probability: {prob:.4f}")
        else:
            print("\nNo accidents detected.")
        
        # Save to CSV if requested
        if args.output:
            import pandas as pd
            df = pd. DataFrame(predictions, columns=['timestamp', 'probability'])
            df.to_csv(args.output, index=False)
            print(f"\nPredictions saved to {args.output}")
    
    elif args. mode == 'realtime': 
        process_video_realtime(args. video, pipeline, args.threshold)


if __name__ == '__main__':
    main()