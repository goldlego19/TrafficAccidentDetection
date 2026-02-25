import cv2
import numpy as np
import os
import glob
import re  # Add this new import

def main(frames_dir):
    # Find all JPG frames in the directory
    frame_files = glob.glob(os.path.join(frames_dir, '*.jpg'))
    
    # --- THE FIX: Sort numerically instead of alphabetically ---
    # This finds the very last number in the filename (e.g., the '4' in '000003_clip_4.jpg')
    # and uses it to sort the files in perfect sequential order
    frame_files = sorted(frame_files, key=lambda f: int(re.findall(r'\d+', os.path.basename(f))[-1]))
    
    if len(frame_files) < 2:
        print(f"Error: Not enough frames found in {frames_dir}")
        return

    # Read the first frame
    frame1 = cv2.imread(frame_files[0])
    
    # ... [keep the rest of the script exactly the same from here down] ...
    if frame1 is None:
        print("Error reading the first frame.")
        return
        
    # Resize for faster processing
    frame1 = cv2.resize(frame1, (640, 360))
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    
    # Create an HSV image to colourise the motion
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255 # Max saturation

    # Loop through the rest of the frames
    for i in range(1, len(frame_files)):
        frame2 = cv2.imread(frame_files[i])
        if frame2 is None:
            continue
            
        frame2 = cv2.resize(frame2, (640, 360))
        next_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Calculate Dense Optical Flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(prvs, next_frame, None, 
                                            pyr_scale=0.5, levels=3, winsize=15, 
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

        # Convert flow vectors to magnitude (speed) and angle (direction)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Map direction to Hue, and speed to Value (brightness)
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        
        # Convert back to BGR to display
        flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # --- ANOMALY DETECTION METRIC ---
        # Count "violent" pixels (pixels moving extremely fast)
        violent_pixels = np.sum(mag > 5.0) 
        
        # Alert logic
        alert = "False"
        colour = (0, 255, 0)
        if violent_pixels > 5000: # Threshold for a crash shockwave
            alert = "TRUE (CRASH DETECTED)"
            colour = (0, 0, 255)

        # Draw metrics on the original frame
        cv2.putText(frame2, f"Violent Pixels: {violent_pixels}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)
        cv2.putText(frame2, f"ALERT: {alert}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)

        # Stack the original frame and the flow heatmap side-by-side
        combined = np.hstack((frame2, flow_bgr))
        
        cv2.imshow('Optical Flow Analysis', combined)
        
        # Press 'q' to quit early (adjust waitKey time to control playback speed)
        if cv2.waitKey(60) & 0xFF == ord('q'):
            break
            
        # Update previous frame
        prvs = next_frame

    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Point this to a folder containing the 50 frames of ONE specific CCTV segment
    frames_directory = "data/cadp/extracted_frames/000003" 
    main(frames_directory)