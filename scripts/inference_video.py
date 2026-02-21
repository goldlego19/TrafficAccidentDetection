import sys
import os
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.accident_detection_model import AccidentDetectionLSTM

# --- CONFIGURATION ---
MODEL_PATH = 'checkpoints/best_model.pth'
YOLO_PATH = 'yolo11m.pt'  # Matches your new training
TARGET_FOLDER = "data/cadp/extracted_frames/000012" # <--- CHECK THIS PATH
OUTPUT_VIDEO = "final_production_output.mp4"
IMG_SIZE = (640, 640)

# --- TUNING  ---
RISK_THRESHOLD = 0.65     # Raised threshold (since model is paranoid)
HORIZON_LINE = 0.30       # Ignore top 40% of screen (Distant overlaps)
MIN_BOX_SIZE = 0.02       # Ignore tiny dots
TRIGGER_FRAMES = 3        # Wait 3 frames to confirm crash
# ---------------------

def enhance_image(img):
    """Matches the CLAHE enhancement used during training"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def calculate_max_iou(boxes, img_height):
    if len(boxes) < 2: return 0.0, -1, -1
    
    max_iou = 0.0
    pair = (-1, -1)
    
    horizon_y = img_height * HORIZON_LINE
    min_h = img_height * MIN_BOX_SIZE

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            boxA = boxes[i]; boxB = boxes[j]
            
            # FILTER 1: Horizon (Fixes Perspective False Positives)
            cyA = (boxA[1] + boxA[3]) / 2
            cyB = (boxB[1] + boxB[3]) / 2
            if cyA < horizon_y and cyB < horizon_y: continue

            # FILTER 2: Size (Fixes Noise)
            if (boxA[3] - boxA[1]) < min_h: continue

            # Calculate IoU
            xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
            interArea = max(0, xB - xA) * max(0, yB - yA)
            
            if interArea == 0: continue
            
            boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
            boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
            iou = interArea / float(boxAArea + boxBArea - interArea)
            
            if iou > max_iou:
                max_iou = iou
                pair = (i, j)
                
    return max_iou, pair[0], pair[1]

def run_production_inference():
    print(f"🎬 Processing: {TARGET_FOLDER}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AccidentDetectionLSTM(hidden_dim=64, num_layers=1, dropout=0.5).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    yolo = YOLO(YOLO_PATH)
    
    folder = Path(TARGET_FOLDER)
    frames = sorted(folder.glob('*.jpg'), key=lambda x: int(x.stem) if x.stem.isdigit() else x.name)
    if not frames: return

    # Video Setup
    first_frame = cv2.imread(str(frames[0]))
    h, w, _ = first_frame.shape
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
    
    buffer = []
    accident_counter = 0

    print("   Settings: CLAHE=On | Horizon Filter=On | Weighted Risk=On")

    for i, frame_path in enumerate(frames):
        img_raw = cv2.imread(str(frame_path))
        if img_raw is None: continue
        
        # 1. ENHANCE (Match Training)
        img_enhanced = enhance_image(img_raw)
        
        # 2. YOLO
        img_resized = cv2.resize(img_enhanced, IMG_SIZE)
        results = yolo(img_resized, conf=0.15, iou=0.5, classes=[2,3,5,7], verbose=False)
        boxes_raw = results[0].boxes.data.cpu().numpy()
        
        # 3. Features
        if len(boxes_raw) > 0:
            boxes_norm = boxes_raw.copy()
            boxes_norm[:, 0] /= IMG_SIZE[1]; boxes_norm[:, 1] /= IMG_SIZE[0]
            boxes_norm[:, 2] /= IMG_SIZE[1]; boxes_norm[:, 3] /= IMG_SIZE[0]
            boxes_norm = boxes_norm[boxes_norm[:, 0].argsort()][:5]
            feat = boxes_norm.flatten()
            feat = np.pad(feat, (0, 128 - len(feat))) if len(feat) < 128 else feat[:128]
        else:
            feat = np.zeros(128)

        buffer.append(feat)
        if len(buffer) > 16: buffer.pop(0)
        
        # 4. AI Prediction
        lstm_prob = 0.0
        if len(buffer) == 16:
            inp = torch.FloatTensor(np.array(buffer)).unsqueeze(0).to(device)
            with torch.no_grad():
                lstm_prob = model(inp).item()

        # 5. Overlap (With Horizon Filter)
        # Pass 'IMG_SIZE[0]' (640) because YOLO boxes are relative to that resize
        current_iou, idxA, idxB = calculate_max_iou(boxes_raw[:, :4], IMG_SIZE[0]) if len(boxes_raw) > 0 else (0.0, -1, -1)

        # 6. Weighted Risk
        # 40% AI + 60% Geometry (Trust physics more than paranoid AI)
        risk_score = (lstm_prob * 0.4) + (current_iou * 0.6)
        
        is_accident = False
        if risk_score > RISK_THRESHOLD:
            is_accident = True
            accident_counter += 1
        else:
            accident_counter = max(0, accident_counter - 1)
            
        final_alert = accident_counter >= TRIGGER_FRAMES

        # --- DRAWING (On Raw Image for Display) ---
        img_display = img_raw.copy()
        
        # Horizon Line (Debug)
        line_y = int(h * HORIZON_LINE)
        cv2.line(img_display, (0, line_y), (w, line_y), (0, 255, 255), 2)

        # Boxes
        for b_idx, box in enumerate(boxes_raw):
            x1, y1, x2, y2 = box[:4].astype(int)
            # Rescale to original size
            x1 = int(x1 * (w / IMG_SIZE[1])); y1 = int(y1 * (h / IMG_SIZE[0]))
            x2 = int(x2 * (w / IMG_SIZE[1])); y2 = int(y2 * (h / IMG_SIZE[0]))
            
            # Check Ignore Status
            cy = (y1 + y2) / 2
            is_ignored = cy < line_y
            
            color = (100, 100, 100) if is_ignored else (0, 255, 0)
            if b_idx in [idxA, idxB] and final_alert: color = (0, 0, 255)
            
            cv2.rectangle(img_display, (x1, y1), (x2, y2), color, 2)

        # Dashboard
        cv2.rectangle(img_display, (0, 0), (w, 60), (0, 0, 0), -1)
        risk_color = (0, 255, 0)
        if risk_score > 0.45: risk_color = (0, 165, 255)
        if risk_score > RISK_THRESHOLD: risk_color = (0, 0, 255)
        
        text = f"Risk: {risk_score:.2f} (AI:{lstm_prob:.2f} IoU:{current_iou:.2f})"
        cv2.putText(img_display, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, risk_color, 2)
        cv2.putText(img_display, "IGNORE ZONE", (5, line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if final_alert:
            cv2.putText(img_display, "ACCIDENT DETECTED", (50, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            cv2.rectangle(img_display, (0, 0), (w, h), (0, 0, 255), 10)

        out.write(img_display)
        if i % 30 == 0: print(f"Processing... Risk: {risk_score:.2f}", end='\r')

    out.release()
    print(f"\n✅ Done! Saved to {OUTPUT_VIDEO}")

if __name__ == "__main__":
    run_production_inference()