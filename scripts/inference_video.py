import sys
import os
import glob
import re
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
# Point this to the ROOT folder containing all your sub-folders (000001, 000002, etc.)
RAW_FRAMES_ROOT = "./data/cadp/extracted_frames" 
OUTPUT_ROOT = "./data/inference_output"
CHECKPOINT_PATH = "./checkpoints/best_resnet_model.pth"

# --- THRESHOLD SETTINGS ---
CONFIDENCE_THRESHOLD = 0.98   
PIXEL_THRESHOLD = 2500        
REQUIRED_FRAMES = 5           
COOLDOWN_FRAMES = 60          

os.makedirs(OUTPUT_ROOT, exist_ok=True)

def process_video(video_folder_path, model, device, transform):
    video_name = os.path.basename(os.path.normpath(video_folder_path))
    video_path = os.path.join(OUTPUT_ROOT, f"final_diagnostic_{video_name}.mp4")
    
    # Get and sort frames
    frame_files = sorted(glob.glob(os.path.join(video_folder_path, '*.jpg')), 
                         key=lambda f: int(re.findall(r'\d+', os.path.basename(f))[-1]))

    if len(frame_files) < 2:
        return f"Skipped {video_name}: Not enough frames."

    # Setup Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Switch back to XVID if you still get .dll errors
    out_video = cv2.VideoWriter(video_path, fourcc, 30.0, (1280, 360))

    # Physics & Trigger Variables
    prvs_gray = cv2.cvtColor(cv2.resize(cv2.imread(frame_files[0]), (640, 360)), cv2.COLOR_BGR2GRAY)
    hsv = np.zeros((360, 640, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    prvs_flow = None
    
    trigger_counter = 0
    cooldown_counter = 0
    is_alarm_active = False

    for i in range(1, len(frame_files)):
        img = cv2.imread(frame_files[i])
        if img is None: continue
        curr_frame = cv2.resize(img, (640, 360))
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        curr_flow = cv2.calcOpticalFlowFarneback(prvs_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        if prvs_flow is not None:
            flow_diff = curr_flow - prvs_flow
            mag, ang = cv2.cartToPolar(flow_diff[..., 0], flow_diff[..., 1])
            
            # --- THE ROI FIX ---
            # Ignores the very bottom of the screen to stop perspective false alarms
            mag[320:, :] = 0 
            
            mag[mag < 3.0] = 0.0 
            
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = np.clip(mag * 10.0, 0, 255).astype(np.uint8)
            flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            input_tensor = transform(Image.fromarray(cv2.cvtColor(flow_bgr, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
            with torch.no_grad():
                prob = model(input_tensor).item()

            bright_pixels = np.sum(hsv[..., 2] > 50)
            instant_hit = (prob > CONFIDENCE_THRESHOLD) and (bright_pixels > PIXEL_THRESHOLD)

            if instant_hit:
                trigger_counter += 1
            else:
                trigger_counter = max(0, trigger_counter - 1)

            if trigger_counter >= REQUIRED_FRAMES:
                is_alarm_active = True
                cooldown_counter = COOLDOWN_FRAMES

            if is_alarm_active:
                cooldown_counter -= 1
                if cooldown_counter <= 0:
                    is_alarm_active = False
                    trigger_counter = 0

            display_frame = curr_frame.copy()
            text_color = (0, 0, 255) if is_alarm_active else (0, 255, 0)
            status = "!! ACCIDENT !!" if is_alarm_active else "Safe"
            
            cv2.putText(display_frame, f"STATUS: {status}", (20, 50), 2, 1.2, text_color, 3)
            cv2.putText(display_frame, f"AI Prob: {prob:.2f} | Pixels: {bright_pixels}", (20, 90), 2, 0.6, (255, 255, 255), 1)

            if is_alarm_active:
                cv2.rectangle(display_frame, (0,0), (640,360), (0,0,255), 15)

            out_video.write(np.hstack((display_frame, flow_bgr)))

        prvs_gray, prvs_flow = curr_gray, curr_flow

    out_video.release()
    return f"Done: {video_name}"

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Initializing Batch Inference on {device}...")
    
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Get list of all folders in the root frames directory
    video_folders = [f.path for f in os.scandir(RAW_FRAMES_ROOT) if f.is_dir()]
    print(f"📂 Found {len(video_folders)} videos to process.")

    for folder in video_folders:
        result = process_video(folder, model, device, transform)
        print(result)

    print("✅ All videos processed!")

if __name__ == '__main__':
    main()