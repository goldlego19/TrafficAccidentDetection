import cv2
import csv
import os

VIDEO_PATH = "TU-DAT/TU-DAT/Final_videos/Positive_Vidoes/v18.mov"
CSV_PATH = "annotations.csv"

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

paused = False
current_frame = 0

accident_start = None
accident_end = None

print("Controls:")
print(" space -> pause / resume")
print(" a     -> step backward (paused)")
print(" d     -> step forward (paused)")
print(" s     -> mark accident START")
print(" e     -> mark accident END")
print(" q/ESC -> save & quit")

def read_frame(frame_idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    return ret, frame

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    else:
        ret, frame = read_frame(current_frame)
        if not ret:
            break

    timestamp = current_frame / fps

    display = frame.copy()
    cv2.putText(display, f"Frame: {current_frame}/{total_frames}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(display, f"Time: {timestamp:.2f}s",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if accident_start is not None:
        cv2.putText(display, f"START: {accident_start:.2f}s",
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    if accident_end is not None:
        cv2.putText(display, f"END: {accident_end:.2f}s",
                    (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Annotate Accident", display)

    key = cv2.waitKey(1 if not paused else 0)

    if key == ord(' '):
        paused = not paused

    elif key == ord('a') and paused:
        current_frame = max(0, current_frame - 1)

    elif key == ord('d') and paused:
        current_frame = min(total_frames - 1, current_frame + 1)

    elif key == ord('s'):
        accident_start = timestamp
        print(f"[START] {accident_start:.2f}s")

    elif key == ord('e'):
        accident_end = timestamp
        print(f"[END] {accident_end:.2f}s")

    elif key in [27, ord('q')]:
        break

cap.release()
cv2.destroyAllWindows()

# ----------------------------
# Save annotation
# ----------------------------
if accident_start is not None:
    if accident_end is None:
        accident_end = total_frames / fps
        print("END not marked — using video end.")

    write_header = not os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["video", "accident_start", "accident_end"])
        writer.writerow([
            os.path.basename(VIDEO_PATH),
            round(accident_start, 2),
            round(accident_end, 2)
        ])
    print("Annotation saved.")
else:
    print("No accident START marked — nothing saved.")
