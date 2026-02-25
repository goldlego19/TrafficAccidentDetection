import cv2
import numpy as np
import os
import glob
import re
import csv
import concurrent.futures

def process_single_video(folder_name, folder_path, start_f, end_f, out_accident, out_normal):
    """Processes a single video folder using Acceleration Optical Flow."""
    # Get and sort frames numerically
    frame_files = glob.glob(os.path.join(folder_path, '*.jpg'))
    frame_files = sorted(frame_files, key=lambda f: int(re.findall(r'\d+', os.path.basename(f))[-1]))

    if len(frame_files) < 2:
        return f"Skipped {folder_name} (not enough frames)"

    # Initialise the first frame
    frame1 = cv2.imread(frame_files[0])
    frame1 = cv2.resize(frame1, (640, 360))
    prvs_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    
    # Create an HSV image to colourise the motion
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255 
    
    # We need to store the previous flow to calculate acceleration
    prvs_flow = None 

    # Process the chronological frames
    # Process the chronological frames
    for i in range(1, len(frame_files)):
        curr_frame = cv2.imread(frame_files[i])
        if curr_frame is None:
            continue
            
        curr_frame = cv2.resize(curr_frame, (640, 360))
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # 1. Calculate the current velocity
        curr_flow = cv2.calcOpticalFlowFarneback(prvs_gray, curr_gray, None, 
                                            pyr_scale=0.5, levels=3, winsize=15, 
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        # 2. Calculate the acceleration (difference in velocity)
        if prvs_flow is not None:
            flow_diff = curr_flow - prvs_flow
            mag, ang = cv2.cartToPolar(flow_diff[..., 0], flow_diff[..., 1])
            
            # --- THE NOISE FLOOR FIX ---
            NOISE_FLOOR = 3.0
            mag[mag < NOISE_FLOOR] = 0.0
            
            hsv[..., 0] = ang * 180 / np.pi / 2
            SENSITIVITY = 10.0 
            hsv[..., 2] = np.clip(mag * SENSITIVITY, 0, 255).astype(np.uint8)
            
            flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            frame_num = int(re.findall(r'\d+', os.path.basename(frame_files[i]))[-1])

            # --- 3. THE DANGER ZONE & UNDERSAMPLING FIX ---
            BUFFER = 150
            skip_saving = False
            save_dir = ""

            if start_f <= frame_num <= end_f:
                save_dir = out_accident
            elif frame_num < (start_f - BUFFER) or frame_num > (end_f + BUFFER):
                # Safely far away from the crash. Apply undersampling.
                if frame_num % 10 != 0:
                    skip_saving = True # Don't save, but DO NOT use 'continue'
                else:
                    save_dir = out_normal
            else:
                # In the Danger Zone! Throw it away.
                skip_saving = True

            # --- 4. THE FOLDER-SPECIFIC SAVE LOGIC ---
            if not skip_saving and save_dir != "":
                base_name = os.path.basename(frame_files[i])
                save_path = os.path.join(save_dir, f"{folder_name}_{base_name}")
                
                bright_pixels = np.sum(hsv[..., 2] > 50)
                
                if save_dir == out_accident:
                    if bright_pixels > 1500: 
                        cv2.imwrite(save_path, flow_bgr)
                        
                elif save_dir == out_normal:
                    if bright_pixels <= 1500:
                        cv2.imwrite(save_path, flow_bgr)

        # --- 5. THE CRITICAL VARIABLE UPDATE ---
        # This MUST happen every single frame to keep the physics accurate!
        prvs_gray = curr_gray
        prvs_flow = curr_flow
        
    return f"Finished {folder_name}"


def main():
    # --- CONFIGURATION ---
    # Point these to wherever your raw sequential CCTV frames are stored
    RAW_FRAMES_ROOT = "./data/cadp/extracted_frames" # Folder containing all your video sub-folders
    OUTPUT_ROOT = "./data/optical_flow_maps"
    CSV_PATH = "./annotations/accidents_cleaned.csv"
    
    out_accident = os.path.join(OUTPUT_ROOT, 'accident')
    out_normal = os.path.join(OUTPUT_ROOT, 'normal')
    
    # Create the directories if they don't exist
    os.makedirs(out_accident, exist_ok=True)
    os.makedirs(out_normal, exist_ok=True)

    # 1. Read the CSV file
    accident_windows = {}
    with open(CSV_PATH, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid_id = row['videoid'] 
            start_f = int(row['startframe'])
            end_f = int(row['endframe'])
            accident_windows[vid_id] = (start_f, end_f)

    print("CSV loaded successfully. Preparing parallel processing...")

    # 2. Build a list of tasks
    tasks = []
    for folder_name in os.listdir(RAW_FRAMES_ROOT):
        folder_path = os.path.join(RAW_FRAMES_ROOT, folder_name)
        if not os.path.isdir(folder_path):
            continue
            
        if folder_name not in accident_windows:
            print(f"Skipping {folder_name}: Not found in CSV.")
            continue
            
        start_f, end_f = accident_windows[folder_name]
        tasks.append((folder_name, folder_path, start_f, end_f, out_accident, out_normal))

    print(f"Starting {len(tasks)} video folders across multiple CPU cores...")

    # 3. Run tasks in parallel using all available CPU cores
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_single_video, *task) for task in tasks]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                print(result)
            except Exception as e:
                print(f"A video processing task failed: {e}")

    print("Dataset generation complete!")

if __name__ == '__main__':
    main()

