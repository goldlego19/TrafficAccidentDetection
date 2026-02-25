import sys
import os
import cv2
import csv
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# Fix path to find src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.influence_map import InfluenceMapGenerator

# --- CONFIG ---
DATA_DIR = Path('data/cadp/extracted_frames')
CSV_FILE = 'annotations/accidents_cleaned.csv'
OUTPUT_DIR = Path('data/influence_maps')
MODEL_PATH = 'cadp_custom_yolo/v1_traffic_model2/weights/best.pt' 
SEQ_LEN = 16
IMG_SIZE = (640, 640)
# --------------

def run():
    # Create dataset directories
    (OUTPUT_DIR / 'accident').mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'normal').mkdir(parents=True, exist_ok=True)
    
    print("🚀 Loading Custom YOLO Tracker...")
    yolo_model = YOLO(MODEL_PATH)
    map_generator = InfluenceMapGenerator(target_size=(224, 224))
    
    tasks = []
    print("📂 Reading Annotations...")
    with open(CSV_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid_id = f"{int(row['videoid']):06d}"
            start = int(row['startframe'])
            end = int(row['endframe'])
            
            folder = DATA_DIR / vid_id
            if not folder.exists(): continue
            frames = sorted(list(folder.glob('*.jpg')), key=lambda x: int(x.stem))
            
            safe_start = max(0, start)
            if safe_start > SEQ_LEN:
                tasks.append({'vid': vid_id, 'type': 'normal', 'frames': frames[max(0, safe_start-SEQ_LEN):safe_start]})
            safe_end = min(len(frames), end)
            if safe_end > safe_start:
                tasks.append({'vid': vid_id, 'type': 'accident', 'frames': frames[safe_start:min(len(frames), safe_start+SEQ_LEN)]})

    print(f"⚡ Generating Influence Maps for {len(tasks)} sequences...")
    
    for i, task in enumerate(tqdm(tasks)):
        if len(task['frames']) != SEQ_LEN:
            continue
            
        track_history = {}
        
        # 1. Tracking Phase
        for p in task['frames']:
            img = cv2.imread(str(p))
            img = cv2.resize(img, IMG_SIZE)
            
            # Run YOLO tracker
            results = yolo_model.track(img, persist=True, classes=[0,1,2,3], verbose=False)
            
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xywh.cpu().numpy()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                
                for box, track_id in zip(boxes, track_ids):
                    if track_id not in track_history:
                        track_history[track_id] = []
                    track_history[track_id].append(box) # [x_center, y_center, width, height]
                    
        # 2. Map Generation Phase
        if len(track_history) > 0:
            inf_map = map_generator.generate_map(track_history, original_size=IMG_SIZE)
            
            # Save the image
            filename = OUTPUT_DIR / task['type'] / f"{task['vid']}_clip_{i}.jpg"
            cv2.imwrite(str(filename), inf_map)

    print(f"✅ Finished! Influence Maps saved to {OUTPUT_DIR}")

if __name__ == '__main__':
    run()