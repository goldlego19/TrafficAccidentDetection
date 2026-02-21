import torch
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms
from collections import deque
from src.video_model import VideoAccidentDetector

def run_inference(frame_folder, model_path, sequence_length=16, buffer_size=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Initializing Inference on {device}...")
    
    # 1. Load the R(2+1)D Model
    model = VideoAccidentDetector(pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device).eval()
    
    # 2. Strict Pre-processing
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
    ])

    frame_dir = Path(frame_folder)
    frame_paths = sorted(list(frame_dir.glob('*.jpg')), key=lambda x: int(x.stem))
    
    if len(frame_paths) < sequence_length:
        print("❌ Not enough frames in the folder to run inference.")
        return

    # 3. Smoothing and History Buffers
    # Buffer size 8 with a step of 2 means we average the last 16 frames of predictions
    prob_buffer = deque(maxlen=buffer_size) 
    history_graph = deque(maxlen=100) # Keeps the last 100 predictions for the UI graph
    
    # Default to 0 for a clean graph start
    for _ in range(100): history_graph.append(0.0)

    print(f"▶️ Playing video: {frame_folder}")
    
    with torch.no_grad():
        # Step by 2 for smooth UI, but rely on the buffer to kill false positives
        for i in range(0, len(frame_paths) - sequence_length, 1):
            segment = frame_paths[i : i + sequence_length]
            tensor_list = []
            display_frame = None
            
            # Load frames
            for idx, p in enumerate(segment):
                img = cv2.imread(str(p))
                if idx == len(segment) - 1:
                    display_frame = img.copy() # Use the most recent frame for the UI
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                tensor_list.append(transform(img_rgb))
            
            # Run model
            input_tensor = torch.stack(tensor_list).permute(1, 0, 2, 3).unsqueeze(0).to(device)
            raw_prob = model(input_tensor).item()
            
            # 4. Apply Temporal Smoothing
            prob_buffer.append(raw_prob)
            smoothed_prob = sum(prob_buffer) / len(prob_buffer)
            history_graph.append(smoothed_prob)
            
            # 5. Advanced UI Overlay
            h, w, _ = display_frame.shape
            overlay = display_frame.copy()
            
            # Threshold set to 0.65 to be strict and ignore stutters
            is_accident = smoothed_prob > 0.65 
            
            # Dynamic Colors: Red for accident, Green for safe
            color = (0, 0, 255) if is_accident else (0, 255, 0)
            status_text = "⚠️ CRITICAL: COLLISION DETECTED" if is_accident else "✔️ SYSTEM NORMAL"
            
            # Top HUD (Semi-transparent)
            cv2.rectangle(overlay, (0, 0), (w, 85), (15, 15, 15), -1)
            display_frame = cv2.addWeighted(overlay, 0.85, display_frame, 0.15, 0)
            
            cv2.putText(display_frame, "Traffic Accident Detection [3D-CNN]", (15, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
            cv2.putText(display_frame, f"STATUS: {status_text}", (15, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
            
            # Bottom Right: Live Probability Graph
            graph_w, graph_h = 250, 80
            graph_x, graph_y = w - graph_w - 20, h - graph_h - 30
            
            # Graph background
            cv2.rectangle(display_frame, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (0, 0, 0), -1)
            cv2.rectangle(display_frame, (graph_x, graph_y), (graph_x + graph_w, graph_y + graph_h), (100, 100, 100), 1)
            
            cv2.putText(display_frame, f"Confidence: {smoothed_prob*100:.1f}%", (graph_x, graph_y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Plot the graph lines
            for j in range(1, len(history_graph)):
                x1 = graph_x + int((j - 1) * (graph_w / 100))
                y1 = graph_y + graph_h - int(history_graph[j - 1] * graph_h)
                x2 = graph_x + int(j * (graph_w / 100))
                y2 = graph_y + graph_h - int(history_graph[j] * graph_h)
                
                line_color = (0, 0, 255) if history_graph[j] > 0.65 else (0, 255, 0)
                cv2.line(display_frame, (x1, y1), (x2, y2), line_color, 2)

            # Red flashing border during an accident
            if is_accident:
                cv2.rectangle(display_frame, (0, 0), (w, h), (0, 0, 255), 8)
                
            # Progress bar
            progress_x = int((i / (len(frame_paths) - sequence_length)) * w)
            cv2.line(display_frame, (0, h-5), (progress_x, h-5), color, 10)

            # Display
            cv2.imshow("Accident Detection System", display_frame)
            
            # WaitKey controls playback speed (15ms is good for 30fps viewing)
            if cv2.waitKey(15) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_inference(
        frame_folder='data/cadp/extracted_frames/000003', 
        model_path='best_temporal_model.pth'
    )