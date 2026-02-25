import cv2
import numpy as np
import math

class InfluenceMapGenerator:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        
    def generate_map(self, track_history, original_size=(640, 640)):
        influence_map = np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
        
        scale_x = self.target_size[0] / original_size[0]
        scale_y = self.target_size[1] / original_size[1]
        
        T_max = max([len(hist) for hist in track_history.values()]) if track_history else 1

        # Pre-calculate intersections for the current frame to identify crashing vehicles
        crashing_ids = set()
        latest_boxes = {}
        for track_id, history in track_history.items():
            if len(history) > 0:
                xc, yc, w, h = history[-1]
                xmin, ymin = (xc - w/2) * scale_x, (yc - h/2) * scale_y
                xmax, ymax = (xc + w/2) * scale_x, (yc + h/2) * scale_y
                latest_boxes[track_id] = (xmin, ymin, xmax, ymax, w * scale_x, h * scale_y)

        # Collision Detection Logic
        # Collision Detection Logic (The Goldilocks IoU Filter)
        # Collision Detection Logic (Kinematic IoU Filter)
        box_ids = list(latest_boxes.keys())
        for i in range(len(box_ids)):
            for j in range(i+1, len(box_ids)):
                id1, id2 = box_ids[i], box_ids[j]
                x1_min, y1_min, x1_max, y1_max, w1, h1 = latest_boxes[id1]
                x2_min, y2_min, x2_max, y2_max, w2, h2 = latest_boxes[id2]
                
                # Check if rectangles physically intersect
                if not (x1_max < x2_min or x2_max < x1_min or y1_max < y2_min or y2_max < y1_min):
                    ix_min, iy_min = max(x1_min, x2_min), max(y1_min, y2_min)
                    ix_max, iy_max = min(x1_max, x2_max), min(y1_max, y2_max)
                    overlap_area = (ix_max - ix_min) * (iy_max - iy_min)
                    
                    area1, area2 = w1 * h1, w2 * h2
                    iou = overlap_area / float(area1 + area2 - overlap_area)
                    
                    # If they overlap in 2D perspective
                    if 0.10 < iou < 0.80:
                        hist1 = track_history[id1]
                        hist2 = track_history[id2]
                        
                        # Helper function to check for violent deceleration
                        def violently_decelerated(hist):
                            if len(hist) < 6: return False
                            # Calculate speed a few frames ago vs speed right now
                            v_old = math.hypot(hist[-3][0] - hist[-6][0], hist[-3][1] - hist[-6][1])
                            v_new = math.hypot(hist[-1][0] - hist[-3][0], hist[-1][1] - hist[-3][1])
                            
                            # True if the car was moving (v_old > 2.0) but suddenly lost 50% of its speed
                            return v_new < (v_old * 0.5) and v_old > 2.0

                        # Only trigger the anomaly if they overlap AND someone suddenly stops
                        if violently_decelerated(hist1) or violently_decelerated(hist2):
                            crashing_ids.add(id1)
                            crashing_ids.add(id2)

        for track_id, history in track_history.items():
            if len(history) < 2:
                continue
                
            for t, box in enumerate(history):
                xc, yc, w, h = box
                xc_s, yc_s = int(xc * scale_x), int(yc * scale_y)
                w_s, h_s = int(w * scale_x), int(h * scale_y)
                
                # 1. BLUE CIRCLES: Paper Equation 1 & 3
                r_c = int(math.sqrt(w_s**2 + h_s**2) / 2) 
                c_depth = min(255, int(4 + (60 / T_max) * (t + 1))) 
                cv2.circle(influence_map, (xc_s, yc_s), r_c, (c_depth, 0, 0), -1)

                # 2. GREEN CIRCLES: Paper Equation 4 logic
                # Drawn AT the trajectory coordinate of the crashing cars
                if t == len(history) - 1 and track_id in crashing_ids:
                    # 'y' radius hardcoded to be larger than the car's blue circle
                    cv2.circle(influence_map, (xc_s, yc_s), int(r_c * 1.5), (0, 255, 0), -1)

            # 3. RED LINES: Corrected trajectories
            for t in range(1, len(history)):
                pt1 = (int(history[t-1][0] * scale_x), int(history[t-1][1] * scale_y))
                pt2 = (int(history[t][0] * scale_x), int(history[t][1] * scale_y))
                cv2.line(influence_map, pt1, pt2, (0, 0, 255), 2)

        return influence_map