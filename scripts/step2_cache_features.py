import sys
import os
import numpy as np
import cv2
import csv
import pickle
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import threading

# Fix path to find src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.accident_detection_model import YOLOFeatureExtractor

# --- CONFIG ---
DATA_DIR = Path('data/cadp/extracted_frames')
CSV_FILE = 'annotations/accidents_cleaned.csv'
OUTPUT_FILE = Path('feature_cache/final_features.pkl')
MODEL_PATH = 'cadp_custom_yolo/v1_traffic_model2/weights/best.pt' 
SEQ_LEN = 16
IMG_SIZE = (640, 640)
# --------------

def enhance_image(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

def process_clip(task, extractor, yolo_lock):
    """Worker function to process a single 16-frame clip."""
    processed_frames = []
    
    # 1. PARALLEL PHASE: I/O and Preprocessing (Runs concurrently)
    for p in task['frames']:
        img = cv2.imread(str(p))
        if img is None: 
            return None
            
        img = enhance_image(img)
        img = cv2.resize(img, IMG_SIZE)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        processed_frames.append(img_rgb)
        
    if len(processed_frames) != SEQ_LEN:
        return None

    # 2. SEQUENTIAL PHASE: GPU Inference (Locked to prevent CUDA crashes)
    feats = []
    with yolo_lock:
        for img_rgb in processed_frames:
            feats.append(extractor.extract_features(img_rgb))
            
    feats_np = np.array(feats)
    
    # Check if the feature array is just empty zeroes
    if np.sum(np.abs(feats_np)) > 0.01:
        return {'features': feats_np, 'label': task['label']}
    return None

def run():
    OUTPUT_FILE.parent.mkdir(exist_ok=True)
    
    print("🚀 Loading YOLO...")
    extractor = YOLOFeatureExtractor(MODEL_PATH)
    yolo_lock = threading.Lock() # Creates the queue system for the GPU
    
    print("📂 Reading Annotations...")
    tasks = []
    with open(CSV_FILE, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid_id = f"{int(row['videoid']):06d}"
            start = int(row['startframe'])
            end = int(row['endframe'])
            
            folder = DATA_DIR / vid_id
            if not folder.exists(): continue
            
            frames = sorted(list(folder.glob('*.jpg')), key=lambda x: int(x.stem))
            
            # Normal Clip
            safe_start = max(0, start)
            if safe_start > SEQ_LEN:
                tasks.append({'frames': frames[max(0, safe_start-SEQ_LEN):safe_start], 'label': 0})
            
            # Accident Clip
            safe_end = min(len(frames), end)
            if safe_end > safe_start:
                tasks.append({'frames': frames[safe_start:min(len(frames), safe_start+SEQ_LEN)], 'label': 1})

    print(f"⚡ Processing {len(tasks)} sequences using Multithreading...")
    
    cache = {}
    valid_count = 0
    
    # Initialise the Thread Pool (8 workers is usually the sweet spot for I/O)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        # Submit all tasks to the pool
        futures = {executor.submit(process_clip, task, extractor, yolo_lock): task for task in tasks}
        
        # Display progress bar as tasks finish
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc="Caching"):
            try:
                result = future.result()
                if result is not None:
                    cache[valid_count] = result
                    valid_count += 1
            except Exception as e:
                print(f"Error processing clip: {e}")

    print(f"\n💾 Saving {valid_count} valid sequences to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'wb') as f:
        pickle.dump(cache, f)

if __name__ == '__main__':
    run()