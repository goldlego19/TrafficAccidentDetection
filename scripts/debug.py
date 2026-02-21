import sys
import os
import cv2
from pathlib import Path
from ultralytics import YOLO

# --- CONFIG ---
YOLO_PATH = 'yolo11n.pt'
TARGET_FOLDER = "data/cadp/extracted_frames/000314" 
OUTPUT_FILE = "debug_yolo_view_final.mp4"
IMG_SIZE = (640, 640) # <--- IMPORTANT
# ----------------

def run_visual_debug():
    print(f"👀 Visualizing YOLO detections (New Settings) in: {TARGET_FOLDER}")
    
    model = YOLO(YOLO_PATH)
    
    folder = Path(TARGET_FOLDER)
    frames = sorted(folder.glob('*.jpg'), key=lambda x: int(x.stem) if x.stem.isdigit() else x.name)
    
    if not frames:
        print("❌ No images found.")
        return

    # Video Writer
    # Note: We resize input to 640x640, so output video will be 640x640
    out = cv2.VideoWriter(OUTPUT_FILE, cv2.VideoWriter_fourcc(*'mp4v'), 30, IMG_SIZE)
    
    for i, frame_path in enumerate(frames):
        img = cv2.imread(str(frame_path))
        if img is None: continue
        
        # Resize to match what the model actually sees
        img_resized = cv2.resize(img, IMG_SIZE)
        
        # Run YOLO with the EXACT settings from your model
        # conf=0.15, classes=[2,3,5,7] (Car, Bike, Bus, Truck)
        results = model(img_resized, conf=0.15, iou=0.5, classes=[2,3,5,7], verbose=False)
        
        # Plot
        res_plotted = results[0].plot()
        
        out.write(res_plotted)
        if i % 30 == 0: print(f"   Frame {i}/{len(frames)}", end='\r')

    out.release()
    print(f"\n✅ Saved visual debug to: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_visual_debug()