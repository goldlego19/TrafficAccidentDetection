"""
Temporal annotation tool with resume support for frames datasets.

Purpose:
- Annotate per-video label (accident/normal) and accident start frame quickly.
- Auto-fill end frame using a chosen policy (e.g., seconds after start or the last frame).
- Persist progress so you can stop anytime and resume later—even mid-video.

Dataset layout (frames):
  cdap/extracted_frames/<video_id>/<frame>.jpg
or any similar root where each subfolder is a video_id containing frame images.

Outputs:
  annotations/videos.csv     -> video_id,num_frames,fps,label
  annotations/accidents.csv  -> video_id,start_frame,end_frame
  annotations/progress.json  -> resume state (last video + frame idx, and completed videos)

Controls:
  j / k       : prev / next frame
  J / K       : prev / next by --step frames (default 5)
  c           : set label = accident
  n           : set label = normal (clears start/end)
  a           : set accident start = current frame number
  e           : set accident end = current frame number (optional manual override)
  Enter       : save annotation for this video and move to next
  s (lowercase): save progress only (no annotation); useful to pause
  q / ESC     : quit; saves progress (video + frame index) so you can resume later

Usage examples:
  python scripts/annotate_temporal_resume.py --frames_root cdap/extracted_frames --fps 30 --auto_end seconds:2.5
  python scripts/annotate_temporal_resume.py --frames_root cdap/extracted_frames --start_from 000245
"""

import os
import cv2
import csv
import json
import argparse
from pathlib import Path

def list_videos(frames_root):
    vids = [d for d in os.listdir(frames_root) if os.path.isdir(os.path.join(frames_root, d))]
    # sort numerically if possible
    def key_fn(x):
        try:
            return int(x)
        except:
            return x
    vids.sort(key=key_fn)
    return vids

def list_frames(vdir):
    frames = [f for f in os.listdir(vdir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    def key_fn(x):
        base = os.path.splitext(x)[0]
        try:
            return int(base)
        except:
            return base
    frames.sort(key=key_fn)
    return frames

def ensure_csv(path, header):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(header)

def read_csv_dict(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))

def upsert_row(path, key_idx, key_val, row_vals, header):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    found = False
    if os.path.exists(path):
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)
            if existing_header is None:
                existing_header = header
            for r in reader:
                if len(r) > key_idx and r[key_idx] == key_val:
                    rows.append(row_vals)
                    found = True
                else:
                    rows.append(r)
    if not found:
        rows.append(row_vals)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

def load_progress(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_progress(path, frames_root, last_video_id=None, last_frame_idx=None, completed=None):
    data = {
        "frames_root": str(frames_root),
        "last_video_id": last_video_id,
        "last_frame_idx": last_frame_idx,
        "completed": sorted(list(completed)) if completed else []
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def parse_auto_end(policy: str, start_frame: int, num_frames: int, fps: float):
    if not policy:
        return None
    policy = policy.strip().lower()
    if policy == "last":
        return num_frames
    if policy.startswith("seconds:"):
        try:
            sec = float(policy.split(":", 1)[1])
        except:
            sec = 2.0
        delta = int(round(sec * (fps if fps and fps > 0 else 30.0)))
        return min(start_frame + delta, num_frames)
    return None

def should_skip_video(vid, videos_rows, accidents_rows):
    # Skip if already in videos.csv with any label; if label==accident, also require accidents.csv entry
    vmap = {r["video_id"]: r for r in videos_rows} if videos_rows else {}
    amap = {r["video_id"]: r for r in accidents_rows} if accidents_rows else {}
    if vid in vmap:
        label = vmap[vid].get("label", "").lower()
        if label == "accident":
            return vid in amap  # skip only if accident window already saved
        else:
            return True  # normal entry saved -> skip
    return False

def annotate_video(frames_root, vid, fps, auto_end, videos_csv, accidents_csv, progress_path, initial_frame_idx=None, step=5):
    vdir = os.path.join(frames_root, vid)
    frames = list_frames(vdir)
    if not frames:
        print(f"[WARN] No frames found for {vid}, skipping.")
        return False, None  # not completed, last_frame_idx unknown

    num_frames = len(frames)
    idx = max(0, min(num_frames - 1, initial_frame_idx or 0))
    label = "accident"  # default for CADP; you can change with 'n'
    start, end = None, None

    videos_header = ["video_id", "num_frames", "fps", "label"]
    accidents_header = ["video_id", "start_frame", "end_frame"]

    print(f"\nAnnotating {vid} ({num_frames} frames)")
    print("Controls: j/k prev/next, J/K prev/next by step, c=accident, n=normal, a=set start, e=set end, Enter=save, s=save progress, q/ESC=quit")

    while True:
        idx = max(0, min(num_frames - 1, idx))
        path = os.path.join(vdir, frames[idx])
        img = cv2.imread(path)
        if img is None:
            img = 255 * (idx % 2 == 0) * (255 // 2)
            img = (img * 0 + 255).astype("uint8")
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        disp = img.copy()

        # derive frame number from filename or positional fallback
        try:
            current_frame_num = int(os.path.splitext(frames[idx])[0])
        except:
            current_frame_num = idx + 1

        # HUD
        cv2.putText(disp, f"{vid} idx={idx+1}/{num_frames} (frame_num={current_frame_num})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(disp, f"label={label} start={start} end={end}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(disp, "j/k prev/next  J/K +/-step  a=start  e=end  Enter=save  s=save progress  q=quit", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

        cv2.imshow("Annotate Accident Start", disp)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord('q'), 27):
            # Save progress and exit
            save_progress(progress_path, frames_root, last_video_id=vid, last_frame_idx=idx, completed=None)
            cv2.destroyAllWindows()
            print(f"[INFO] Quit requested. Progress saved at video={vid}, frame_idx={idx}.")
            return False, idx  # not completed, resume from idx next time

        elif key == ord('s'):
            # Save progress only, continue annotating
            save_progress(progress_path, frames_root, last_video_id=vid, last_frame_idx=idx, completed=None)
            print(f"[INFO] Progress saved at video={vid}, frame_idx={idx}.")

        elif key == ord('j'):
            idx -= 1
        elif key == ord('k'):
            idx += 1
        elif key == ord('J'):
            idx -= step
        elif key == ord('K'):
            idx += step

        elif key == ord('a'):
            start = current_frame_num
            # auto end on setting start if policy present
            end = parse_auto_end(auto_end, start, num_frames, fps)

        elif key == ord('e'):
            end = current_frame_num
            if start is not None and end is not None and end < start:
                start, end = end, start

        elif key in (10, 13):  # Enter
            # finalize end if accident and start present
            label="accident"
            if label == "accident" and start is not None and (end is None or end < start):
                end = parse_auto_end(auto_end, start, num_frames, fps) or num_frames

            # write videos.csv
            upsert_row(videos_csv, 0, vid, [vid, str(num_frames), f"{fps:.3f}", label], videos_header)

            # write accidents.csv only if accident and start present
            if label == "accident" and start is not None:
                upsert_row(accidents_csv, 0, vid, [vid, str(start), str(end if end is not None else start)], accidents_header)

            # clear progress for this video (it is completed)
            save_progress(progress_path, frames_root, last_video_id=None, last_frame_idx=None, completed=None)
            cv2.destroyAllWindows()
            print(f"[SAVE] {vid}: label={label}, start={start}, end={end}")
            return True, None  # completed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_root", default="./data/cadp/extracted_frames", help="Root of frames dataset")
    ap.add_argument("--videos_csv", default="./annotations/videos.csv")
    ap.add_argument("--accidents_csv", default="./annotations/accidents.csv")
    ap.add_argument("--progress_json", default="./annotations/progress.json")
    ap.add_argument("--fps", type=float, default=30.0, help="Default FPS (used for seconds->frames if needed)")
    ap.add_argument("--auto_end", type=str, default="last", help='Auto end policy: "last" or "seconds:<float>"')
    ap.add_argument("--step", type=int, default=5, help="Frame step for J/K keys")
    ap.add_argument("--start_from", type=str, default=None, help="Optional video_id to start from (overrides progress)")
    args = ap.parse_args()

    # Setup CSVs
    ensure_csv(args.videos_csv, ["video_id", "num_frames", "fps", "label"])
    ensure_csv(args.accidents_csv, ["video_id", "start_frame", "end_frame"])

    videos_rows = read_csv_dict(args.videos_csv)
    accidents_rows = read_csv_dict(args.accidents_csv)

    # Load progress
    progress = load_progress(args.progress_json)
    progress_vid = progress.get("last_video_id")
    progress_frame_idx = progress.get("last_frame_idx")

    # List videos
    vids = list_videos(args.frames_root)
    if not vids:
        print(f"[ERROR] No videos found under {args.frames_root}")
        return

    # Determine starting index
    start_vid = args.start_from or progress_vid
    start_idx = 0
    if start_vid and start_vid in vids:
        start_idx = vids.index(start_vid)
    else:
        # find first unannotated video
        for i, vid in enumerate(vids):
            if not should_skip_video(vid, videos_rows, accidents_rows):
                start_idx = i
                break

    print(f"[INFO] Starting at video index {start_idx} ({vids[start_idx]})")
    # Iterate through videos from start_idx
    i = start_idx
    while i < len(vids):
        vid = vids[i]

        # Skip already annotated
        if should_skip_video(vid, videos_rows, accidents_rows):
            i += 1
            continue

        # Initial frame index: from progress if we are resuming same video
        init_idx = progress_frame_idx if progress_vid == vid else None

        completed, _last_frame_idx = annotate_video(
            args.frames_root, vid, args.fps, args.auto_end, args.videos_csv, args.accidents_csv,
            args.progress_json, initial_frame_idx=init_idx, step=args.step
        )

        # Refresh rows (in case we saved)
        videos_rows = read_csv_dict(args.videos_csv)
        accidents_rows = read_csv_dict(args.accidents_csv)

        if not completed:
            # User quit; progress already saved inside annotate_video
            print("[INFO] Exiting by user request.")
            return

        # Move to next video
        i += 1

    print("[DONE] All videos processed or skipped (already annotated).")

if __name__ == "__main__":
    main()
    