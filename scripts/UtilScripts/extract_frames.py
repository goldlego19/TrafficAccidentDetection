import cv2
import os
import argparse

def extract_frames(video_path, output_root):
    # 1. Get the video name to create a specific sub-folder
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder = os.path.join(output_root, video_name)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 2. Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_count = 0
    print(f"Starting extraction for: {video_name}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 3. Save frame as JPG (using high quality)
        save_path = os.path.join(output_folder, f"{frame_count}.jpg")
        cv2.imwrite(save_path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"  Extracted {frame_count} frames...")

    cap.release()
    print(f"Finished! {frame_count} frames saved to: {output_folder}")

if __name__ == "__main__":
    VIDEO_FILE = "./scripts/marsaBridge.mp4"
    OUTPUT_DIR = "./data/mbridge/"
    
    extract_frames(VIDEO_FILE, OUTPUT_DIR)