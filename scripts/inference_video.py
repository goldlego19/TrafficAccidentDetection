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
RAW_FRAMES_ROOT = "./data/mBridge/" 
OUTPUT_ROOT = "./data/inference_outputtest"
CHECKPOINT_PATH = "./checkpoints/best_resnet_model2.pth"

# --- THE DUAL-KEY THRESHOLDS ---
CONFIDENCE_THRESHOLD = 0.80   # AI must be 98% sure it's a crash
PIXEL_THRESHOLD = 2500        # Must be a massive explosion of kinetic energy
REQUIRED_FRAMES = 5           # Must see violence for 5 consecutive frames
COOLDOWN_FRAMES = 60          # Keep the alarm active for 2 seconds (at 30fps)

os.makedirs(OUTPUT_ROOT, exist_ok=True)

def process_video(video_folder_path, model, device, transform):
    """Runs physics extraction and AI inference on a single CCTV video."""
    video_name = os.path.basename(os.path.normpath(video_folder_path))
    video_path = os.path.join(OUTPUT_ROOT, f"final_diagnostic_{video_name}.mp4")
    
    # 1. Load and sort the chronological frames
    frame_files = sorted(glob.glob(os.path.join(video_folder_path, '*.jpg')), 
                         key=lambda f: int(re.findall(r'\d+', os.path.basename(f))[-1]))

    if len(frame_files) < 2:
        return f"Skipped {video_name}: Not enough frames."

    # 2. Setup the output video builder (using XVID for wide compatibility)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out_video = cv2.VideoWriter(video_path, fourcc, 30.0, (1280, 360))

    # 3. Initialise the physics variables
    first_frame = cv2.resize(cv2.imread(frame_files[0]), (640, 360))
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    
    hsv = np.zeros((360, 640, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    prev_flow = None
    
    # 4. Initialise the alarm logic
    trigger_counter = 0
    cooldown_counter = 0
    is_alarm_active = False

    # Process frame by frame
    for i in range(1, len(frame_files)):
        img = cv2.imread(frame_files[i])
        if img is None: continue
        
        curr_frame = cv2.resize(img, (640, 360))
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # --- PHASE A: CALCULATE VELOCITY ---
        curr_flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 
                                                 0.5, 3, 15, 3, 5, 1.2, 0)

        # --- PHASE B: CALCULATE ACCELERATION ---
        if prev_flow is not None:
            flow_diff = curr_flow - prev_flow
            magnitude, angle = cv2.cartToPolar(flow_diff[..., 0], flow_diff[..., 1])
            
            # --- PHASE C: PHYSICAL FILTERS ---
            # 1. ROI Mask: Ignore the bottom of the screen (Perspective Trap)
            magnitude[320:, :] = 0 
            
            # 2. Noise Floor: Ignore tiny sensor vibrations
            magnitude[magnitude < 3.0] = 0.0 
            
            # Colourise the surviving physics data
            hsv[..., 0] = angle * 180 / np.pi / 2
            hsv[..., 2] = np.clip(magnitude * 10.0, 0, 255).astype(np.uint8)
            flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # --- PHASE D: AI PREDICTION ---
            # Send the physics heatmap to the ResNet18 brain
            input_tensor = transform(Image.fromarray(cv2.cvtColor(flow_bgr, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
            with torch.no_grad():
                ai_probability = model(input_tensor).item()

            # --- PHASE E: DUAL-KEY ALARM LOGIC ---
            bright_pixels = np.sum(hsv[..., 2] > 50)
            
            # Both the AI and the Physics must agree!
            instant_hit = (ai_probability > CONFIDENCE_THRESHOLD) and (bright_pixels > PIXEL_THRESHOLD)

            # Require sustained violence to trigger the alarm
            if instant_hit:
                trigger_counter += 1
            else:
                trigger_counter = max(0, trigger_counter - 1)

            if trigger_counter >= REQUIRED_FRAMES:
                is_alarm_active = True
                cooldown_counter = COOLDOWN_FRAMES

            # Keep the alarm visible even if the cars stop moving
            if is_alarm_active:
                cooldown_counter -= 1
                if cooldown_counter <= 0:
                    is_alarm_active = False
                    trigger_counter = 0

            # --- PHASE F: VISUALISATION & SAVING ---
            display_frame = curr_frame.copy()
            text_color = (0, 0, 255) if is_alarm_active else (0, 255, 0)
            status = "!! ACCIDENT !!" if is_alarm_active else "Safe"
            
            cv2.putText(display_frame, f"STATUS: {status}", (20, 50), 2, 1.2, text_color, 3)
            cv2.putText(display_frame, f"AI Prob: {ai_probability:.2f} | Pixels: {bright_pixels}", 
                        (20, 90), 2, 0.6, (255, 255, 255), 1)

            if is_alarm_active:
                cv2.rectangle(display_frame, (0,0), (640,360), (0,0,255), 15)

            # Stitch the raw camera and the physics view together side-by-side
            out_video.write(np.hstack((display_frame, flow_bgr)))

        # Always update history for the next frame
        prev_gray = curr_gray
        prev_flow = curr_flow

    out_video.release()
    return f"Done: {video_name}"


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initialising Batch Inference on {device}...")
    
    # 1. Load the frozen AI Brain
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

    # 2. Discover all video folders
    video_folders = [f.path for f in os.scandir(RAW_FRAMES_ROOT) if f.is_dir()]
    print(f"Found {len(video_folders)} videos to process.")

    # 3. Process sequentially
    for folder in video_folders:
        result = process_video(folder, model, device, transform)
        print(result)

    print("All videos processed!")

if __name__ == '__main__':
    main()