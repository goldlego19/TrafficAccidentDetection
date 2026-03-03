import cv2
import numpy as np
import os
import glob
import re
import csv
import concurrent.futures

def process_single_video(folder_name, folder_path, crash_start_frame, crash_end_frame, out_accident, out_normal):
    """Extracts acceleration physics from a video and saves the visual heatmaps."""
    
    # 1. Load and sort the frames numerically
    frame_files = glob.glob(os.path.join(folder_path, '*.jpg'))
    frame_files = sorted(frame_files, key=lambda f: int(re.findall(r'\d+', os.path.basename(f))[-1]))

    if len(frame_files) < 2:
        return f"Skipped {folder_name} (not enough frames)"

    # Setup the initial physics variables
    frame1 = cv2.resize(cv2.imread(frame_files[0]), (640, 360))
    prvs_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255 
    prvs_flow = None 

    # Process frame by frame
    for i in range(1, len(frame_files)):
        curr_frame = cv2.resize(cv2.imread(frame_files[i]), (640, 360))
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # --- STEP 1: CALCULATE VELOCITY ---
        curr_flow = cv2.calcOpticalFlowFarneback(prvs_gray, curr_gray, None, 
                                                 0.5, 3, 15, 3, 5, 1.2, 0)

        # --- STEP 2: CALCULATE ACCELERATION ---
        if prvs_flow is not None:
            flow_diff = curr_flow - prvs_flow
            mag, ang = cv2.cartToPolar(flow_diff[..., 0], flow_diff[..., 1])
            
            # Filter out tiny camera vibrations
            mag[mag < 3.0] = 0.0
            
            # --- STEP 3: COLOURISE THE PHYSICS ---
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = np.clip(mag * 10.0, 0, 255).astype(np.uint8)
            flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # --- STEP 4: CATEGORISE THE FRAME ---
            frame_num = int(re.findall(r'\d+', os.path.basename(frame_files[i]))[-1])
            BUFFER = 150
            category = "ignore" # Default to ignoring the frame

            is_crashing = crash_start_frame <= frame_num <= crash_end_frame
            is_danger_zone = (crash_start_frame - BUFFER) < frame_num < (crash_end_frame + BUFFER)

            if is_crashing:
                category = "accident"
            elif not is_danger_zone:
                category = "normal" 

            # --- STEP 5: APPLY PHYSICAL RULES & SAVE ---
            bright_pixels = np.sum(hsv[..., 2] > 50)
            base_name = os.path.basename(frame_files[i])
            
            if category == "accident" and bright_pixels > 1500:
                save_path = os.path.join(out_accident, f"{folder_name}_{base_name}")
                cv2.imwrite(save_path, flow_bgr)
                
            elif category == "normal" and bright_pixels <= 1500:
                save_path = os.path.join(out_normal, f"{folder_name}_{base_name}")
                cv2.imwrite(save_path, flow_bgr)

        # Always update history for the next frame
        prvs_gray = curr_gray
        prvs_flow = curr_flow
        
    return f"Finished {folder_name}"

def main():
    RAW_FRAMES_ROOT = "./data/cadp/extracted_frames" 
    OUTPUT_ROOT = "./data/optical_flow_mapsNEW"
    CSV_PATH = "./annotations/accidents_cleaned.csv"
    
    out_accident = os.path.join(OUTPUT_ROOT, 'accident')
    out_normal = os.path.join(OUTPUT_ROOT, 'normal')
    
    os.makedirs(out_accident, exist_ok=True)
    os.makedirs(out_normal, exist_ok=True)

    # 1. Load Annotation Data
    accident_windows = {}
    with open(CSV_PATH, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            accident_windows[row['videoid']] = (int(row['startframe']), int(row['endframe']))

    # 2. Package tasks for multiprocessing
    tasks = []
    for folder_name in os.listdir(RAW_FRAMES_ROOT):
        folder_path = os.path.join(RAW_FRAMES_ROOT, folder_name)
        
        if os.path.isdir(folder_path) and folder_name in accident_windows:
            crash_start_frame, crash_end_frame = accident_windows[folder_name]
            tasks.append((folder_name, folder_path, crash_start_frame, crash_end_frame, out_accident, out_normal))

    print(f"Processing {len(tasks)} videos across multiple CPU cores...")

    # 3. Execute in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_single_video, *task) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            print(future.result())

    print("Dataset generation complete!")

if __name__ == '__main__':
    main()