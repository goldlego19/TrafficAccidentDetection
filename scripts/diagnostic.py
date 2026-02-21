import sys
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.accident_detection_model import AccidentDetectionLSTM

# --- CONFIGURATION ---
MODEL_PATH = 'checkpoints/best_model.pth'
YOLO_PATH = 'yolo11n.pt'
TARGET_FOLDER = "data/cadp/extracted_frames/000003"   # <--- CHECK YOUR FOLDER
OUTPUT_FILE = "diagnostic_output.mp4"
IMG_SIZE = (640, 640)
# ---------------------

class DiagnosticExtractor:
    """Custom extractor that returns BOTH features AND visual boxes"""
    def __init__(self, model_path, feature_dim=128):
        self.model = YOLO(model_path)
        self.feature_dim = feature_dim

    def process(self, img):
        # Run YOLO with the EXACT settings used in training
        results = self.model(img, conf=0.15, iou=0.5, classes=[2,3,5,7], verbose=False)
        result = results[0]
        
        # 1. Get Boxes for Visualization
        plot_img = result.plot()
        
        # 2. Extract Features for LSTM
        boxes = result.boxes.data.cpu().numpy()
        if len(boxes) == 0:
            return plot_img, torch.zeros(self.feature_dim).numpy(), False

        # Normalize & Sort (Exact same logic as training)
        h, w, _ = img.shape
        boxes[:, 0] /= w; boxes[:, 1] /= h
        boxes[:, 2] /= w; boxes[:, 3] /= h
        boxes = boxes[boxes[:, 0].argsort()][:5] # Left-to-right sort
        
        features = boxes.flatten()
        if len(features) < self.feature_dim:
            features = torch.tensor(features)
            features = torch.nn.functional.pad(features, (0, self.feature_dim - len(features)))
        else:
            features = torch.tensor(features[:self.feature_dim])
            
        return plot_img, features.numpy(), True

def run_diagnostic():
    print(f"🕵️  Running Diagnostic on: {TARGET_FOLDER}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load LSTM
    model = AccidentDetectionLSTM(hidden_dim=64, num_layers=1, dropout=0.5).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    # Load YOLO Helper
    extractor = DiagnosticExtractor(YOLO_PATH)
    
    folder = Path(TARGET_FOLDER)
    frames = sorted(folder.glob('*.jpg'), key=lambda x: int(x.stem) if x.stem.isdigit() else x.name)
    if not frames:
        print("❌ No images found.")
        return

    # Setup Video
    first_frame = cv2.imread(str(frames[0]))
    h, w, _ = first_frame.shape
    out = cv2.VideoWriter(OUTPUT_FILE, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
    
    buffer = []
    
    print(f"   Processing {len(frames)} frames...")
    
    for i, frame_path in enumerate(frames):
        img = cv2.imread(str(frame_path))
        if img is None: continue
        
        # Resize for Model
        img_in = cv2.resize(img, IMG_SIZE)
        
        # 1. Run YOLO & Get Visualization
        visual_frame, feat, has_cars = extractor.process(img_in)
        
        # Resize visual frame back to original size for video output
        visual_frame = cv2.resize(visual_frame, (w, h))
        
        # 2. Run LSTM
        buffer.append(feat)
        if len(buffer) > 16: buffer.pop(0)
        
        prob = 0.0
        if len(buffer) == 16:
            inp = torch.FloatTensor(np.array(buffer)).unsqueeze(0).to(device)
            with torch.no_grad():
                prob = model(inp).item()
        
        # --- DASHBOARD ---
        # Draw Dashboard Background
        cv2.rectangle(visual_frame, (0, 0), (w, 120), (0, 0, 0), -1)
        
        # Draw Probability Bar
        bar_color = (0, 255, 0)
        if prob > 0.60: bar_color = (0, 165, 255) # Orange
        if prob > 0.75: bar_color = (0, 0, 255)   # Red
        
        cv2.putText(visual_frame, f"Crash Prob: {prob:.2f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Bar
        cv2.rectangle(visual_frame, (20, 60), (w-20, 90), (50, 50, 50), -1)
        bar_w = int((w-40) * prob)
        if bar_w > 0:
            cv2.rectangle(visual_frame, (20, 60), (20+bar_w, 90), bar_color, -1)
            
        # Draw Status
        status = "CARS DETECTED" if has_cars else "NO CARS (ZERO INPUT)"
        status_color = (0, 255, 0) if has_cars else (0, 0, 255)
        cv2.putText(visual_frame, f"YOLO: {status}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        out.write(visual_frame)
        if i % 30 == 0: print(f"   Frame {i}/{len(frames)} | Prob: {prob:.2f}", end='\r')

    out.release()
    print(f"\n✅ Diagnostic Video Saved: {OUTPUT_FILE}")
    print("   -> Open this video. If bars are RED but prob is low, threshold is issue.")
    print("   -> If YOLO status is 'NO CARS', input quality is issue.")

if __name__ == "__main__":
    run_diagnostic()