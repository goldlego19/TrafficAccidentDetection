import cv2
import numpy as np
import os
import glob
import re

def main():
    # --- CONFIGURATION ---
    RAW_NORMAL_ROOT = "./data/TAD/frames/normal" 
    OUTPUT_ROOT = "./data/optical_flow_maps_TAD/normal"
    TARGET_FRAMES = 19200  # Set the exact number of frames you want here!
    
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    extracted_count = 0
    
    # Get all video folders
    if not os.path.exists(RAW_NORMAL_ROOT):
        print(f"❌ Directory not found: {RAW_NORMAL_ROOT}")
        return
        
    video_folders = [f for f in os.listdir(RAW_NORMAL_ROOT) if os.path.isdir(os.path.join(RAW_NORMAL_ROOT, f))]
    
    print(f"🎯 Target set: Extracting exactly {TARGET_FRAMES} normal frames...")
    
    for folder_name in video_folders:
        # Check if we hit the target before starting a new video
        if extracted_count >= TARGET_FRAMES:
            break 
            
        folder_path = os.path.join(RAW_NORMAL_ROOT, folder_name)
        frame_files = glob.glob(os.path.join(folder_path, '*.jpg'))
        frame_files = sorted(frame_files, key=lambda f: int(re.findall(r'\d+', os.path.basename(f))[-1]))

        if len(frame_files) < 2:
            continue

        # Setup initial physics
        frame1 = cv2.resize(cv2.imread(frame_files[0]), (640, 360))
        prvs_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255 
        prvs_flow = None 

        for i in range(1, len(frame_files)):
            # Check if we hit the target mid-video
            if extracted_count >= TARGET_FRAMES:
                break 
                
            curr_frame = cv2.resize(cv2.imread(frame_files[i]), (640, 360))
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            curr_flow = cv2.calcOpticalFlowFarneback(prvs_gray, curr_gray, None, 
                                                     0.5, 3, 15, 3, 5, 1.2, 0)

            if prvs_flow is not None:
                flow_diff = curr_flow - prvs_flow
                mag, ang = cv2.cartToPolar(flow_diff[..., 0], flow_diff[..., 1])
                
                # Filter out tiny camera vibrations
                mag[mag < 3.0] = 0.0
                
                hsv[..., 0] = ang * 180 / np.pi / 2
                hsv[..., 2] = np.clip(mag * 10.0, 0, 255).astype(np.uint8)
                flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                
                bright_pixels = np.sum(hsv[..., 2] > 50)
                base_name = os.path.basename(frame_files[i])
                
                # If the physics show it's a safe normal frame, save it
                if bright_pixels <= 1500:
                    save_path = os.path.join(OUTPUT_ROOT, f"{folder_name}_{base_name}")
                    cv2.imwrite(save_path, flow_bgr)
                    
                    extracted_count += 1
                    
                    # Print progress update every 500 frames
                    if extracted_count % 500 == 0:
                        print(f"✅ Extracted {extracted_count}/{TARGET_FRAMES} frames...")

            # Update history for the next frame
            prvs_gray = curr_gray
            prvs_flow = curr_flow

    print(f"\n🎉 Complete! Successfully generated exactly {extracted_count} optical flow maps.")

if __name__ == '__main__':
    main()