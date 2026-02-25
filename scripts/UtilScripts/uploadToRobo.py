"""
Multithreaded Roboflow "1+1" Targeted Uploader for CADP
Uses concurrent API requests to drastically speed up image uploads.
"""
import os
import random
import pandas as pd
from pathlib import Path
from roboflow import Roboflow
import concurrent.futures

def upload_single_frame(project, frame_path, vid_name, frame_type):
    """Worker function to handle a single upload in a separate thread."""
    try:
        project.upload(str(frame_path))
        print(f"  -> ✅ Uploaded ({frame_type}): {vid_name}/{frame_path.name}")
        return True
    except Exception as e:
        print(f"  -> ❌ Failed to upload {frame_path.name}: {e}")
        return False

def upload_1_plus_1_subset_fast(api_key, workspace_name, project_name, data_dir, csv_path, max_workers=8):
    print("Authorising Roboflow connection...")
    rf = Roboflow(api_key=api_key)
    
    try:
        project = rf.workspace(workspace_name).project(project_name)
    except Exception as e:
        print(f"❌ Failed to connect to Roboflow Project: {e}")
        return

    frames_dir = Path(data_dir)
    
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip() 
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    print(f"📂 Found {len(df)} entries. Building the upload queue...")
    
    # 1. Build a list of all frames to upload first
    upload_queue = []
    
    for index, row in df.iterrows():
        try:
            vid_id = str(row['videoid']).zfill(6)
            start_f = int(row['startframe'])
            end_f = int(row['endframe'])
        except KeyError:
            continue
            
        vid_folder = frames_dir / vid_id
        if not vid_folder.exists():
            continue
            
        frame_paths = sorted(list(vid_folder.glob('*.jpg')), key=lambda x: int(x.stem))
        if not frame_paths:
            continue
            
        random_frame = random.choice(frame_paths)
        
        start_idx = min(start_f, len(frame_paths) - 1)
        end_idx = min(end_f, len(frame_paths) - 1)
        
        if start_idx < end_idx:
            crash_idx = (start_idx + end_idx) // 2
            crash_frame = frame_paths[crash_idx]
        else:
            crash_frame = random_frame

        # Add to our queue (handling duplicates if the crash frame equals the random frame)
        frames_to_upload = list(set([random_frame, crash_frame]))
        
        for frame_path in frames_to_upload:
            frame_type = "💥 Crash" if frame_path == crash_frame else "🚗 Normal"
            upload_queue.append((project, frame_path, vid_folder.name, frame_type))

    print(f"🚀 Queue built with {len(upload_queue)} images. Starting multithreaded upload with {max_workers} workers...")
    
    # 2. Execute uploads concurrently
    total_uploaded = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the thread pool
        futures = [executor.submit(upload_single_frame, *task) for task in upload_queue]
        
        # Tally up the successful uploads as they complete
        for future in concurrent.futures.as_completed(futures):
            if future.result():
                total_uploaded += 1

    print(f"\n🎉 Fast Upload Complete! Successfully added {total_uploaded} targeted images to Roboflow.")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    API_KEY = "LMO8yxRpPqtHdGZlRrGM"
    WORKSPACE = "pose-estimation-datasets"   
    PROJECT = "traffic-collision-data-2uw2y"       
    DATA_DIRECTORY = "./data/cadp/extracted_frames"
    CSV_PATH = "./annotations/accidents_cleaned.csv"
    
    upload_1_plus_1_subset_fast(
        api_key=API_KEY,
        workspace_name=WORKSPACE,
        project_name=PROJECT,
        data_dir=DATA_DIRECTORY,
        csv_path=CSV_PATH,
        max_workers=10
    )