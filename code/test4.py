from ultralytics import YOLO
import torch
import cv2
import os

# -----------------------------
# 1. Run YOLO detection on video
# -----------------------------
model = YOLO("yolo11n.pt")
video_path = "videos/soccer1.mp4"
results = model(video_path, save=True, show=True)

# Open the video using OpenCV
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps

# -----------------------------
# 2. Find ball detection timestamps
# -----------------------------
ball_class_name = "sports ball"
ball_class_id = None
for class_id, name in model.names.items():
    if name.lower() == ball_class_name.lower():
        ball_class_id = class_id
        break

if ball_class_id is None:
    print("The 'sports ball' class was not found in the model's classes!")
    cap.release()
    exit()

ball_timestamps = []
for frame_idx, result in enumerate(results):
    if hasattr(result, 'boxes') and len(result.boxes) > 0:
        try:
            cls_ids = result.boxes.cls.cpu().numpy() if isinstance(result.boxes.cls, torch.Tensor) else result.boxes.cls
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            continue

        if ball_class_id in cls_ids:
            timestamp = frame_idx / fps
            ball_timestamps.append(timestamp)

print("Ball detection timestamps (in seconds):", ball_timestamps)
if not ball_timestamps:
    print("No ball detections found in the video.")
    cap.release()
    exit()

# -----------------------------
# 3. Group detections into continuous segments (merging detections within 2 seconds)
# -----------------------------
segments = []
group_start = ball_timestamps[0]
group_end = ball_timestamps[0]

for t in ball_timestamps[1:]:
    if t - group_end <= 2:
        group_end = t
    else:
        segments.append((group_start, group_end))
        group_start = t
        group_end = t
segments.append((group_start, group_end))

# -----------------------------
# 4. Add a 3-second buffer before and after each segment
# -----------------------------
highlight_segments = []
for start, end in segments:
    highlight_start = max(0, start - 3)
    highlight_end = min(video_duration, end + 3)
    highlight_segments.append((highlight_start, highlight_end))

print("Highlight segments (with buffers):", highlight_segments)

# -----------------------------
# 5. Save each highlight as a separate video file using OpenCV
# -----------------------------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
for i, (start, end) in enumerate(highlight_segments):
    output_path = f"highlight_{i+1}.mp4"
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"Saving highlight {i+1}: from {start:.2f} to {end:.2f} seconds.")

    start_frame = int(start * fps)
    end_frame = int(end * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)
    writer.release()

cap.release()
