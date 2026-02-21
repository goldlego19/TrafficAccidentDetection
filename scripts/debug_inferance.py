import sys
import os
import cv2
import torch
import numpy as np
from pathlib import Path

# Fix Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.accident_detection_model import AccidentDetectionLSTM, YOLOFeatureExtractor

# --- CONFIG ---
MODEL_PATH = 'checkpoints/best_model.pth'
YOLO_PATH = 'yolo11n.pt'
TARGET_FOLDER = "data/cadp/extracted_frames/000314"  # <--- MAKE SURE THIS IS A VALID FOLDER
# ----------------

def run_debug():
    print(f"🕵️ DEBUGGING FOLDER: {TARGET_FOLDER}")
    
    # 1. Load YOLO Only (LSTM comes later)
    print("   Loading YOLO...")
    extractor = YOLOFeatureExtractor(YOLO_PATH)
    
    # 2. Get Frames
    folder = Path(TARGET_FOLDER)
    frames = sorted(folder.glob('*.jpg'), key=lambda x: int(x.stem) if x.stem.isdigit() else x.name)
    
    if not frames:
        print("❌ NO FRAMES FOUND! Check your path.")
        return

    print(f"   Found {len(frames)} frames. Checking the first 20...")
    
    # 3. Check what YOLO produces
    zero_detections = 0
    
    for i, frame_path in enumerate(frames[:30]): # Check first 30 frames
        img = cv2.imread(str(frame_path))
        if img is None: continue
        
        # Preprocess (MUST MATCH TRAINING)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Extract
        feat = extractor.extract_features(img)
        
        # Analyze
        feat_sum = np.sum(np.abs(feat))
        is_zero = feat_sum < 0.001
        
        status = "❌ EMPTY (0.0)" if is_zero else f"✅ Data ({feat_sum:.2f})"
        print(f"   Frame {i}: {status}")
        
        if is_zero:
            zero_detections += 1

    print("-" * 30)
    if zero_detections > 10:
        print("⚠️  CRITICAL ISSUE: YOLO is not detecting anything!")
        print("   Reason 1: The images are too dark or blurry.")
        print("   Reason 2: There are no cars in these frames.")
        print("   Reason 3: The resize (224x224) makes the cars too small to see.")
    else:
        print("✅ YOLO is working correctly. The issue is in the LSTM or Model Weights.")
        print("   Proceed to Step 2 below.")

if __name__ == "__main__":
    run_debug()