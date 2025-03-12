from ultralytics import YOLO
import torch
import cv2
import os

# -----------------------------
# Create output directories
# -----------------------------
output_dirs = ["full_detection", "ball_timestamp", "highlights"]
for folder in output_dirs:
    os.makedirs(folder, exist_ok=True)

# -----------------------------
# 1. Run YOLO detection on video
# -----------------------------
model = YOLO("yolo11n.pt")
video_path = "videos/soccer1.mp4"
results = model(video_path, save=True, show=True)

# Open the video using OpenCV and get properties
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_duration = video_frame_count / fps

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
# 4. Add a 3-second buffer before and after each segment for highlights
# -----------------------------
highlight_segments = []
for start, end in segments:
    highlight_start = max(0, start - 3)
    highlight_end = min(video_duration, end + 3)
    highlight_segments.append((highlight_start, highlight_end))

print("Highlight segments (with buffers):", highlight_segments)

# -----------------------------
# 5a. Save ball detection timestamps to a text file
# -----------------------------
timestamp_file = os.path.join("ball_timestamp", "ball_timestamps.txt")
with open(timestamp_file, "w") as f:
    f.write("Raw Ball Detection Timestamps (s):\n")
    for ts in ball_timestamps:
        f.write(f"{ts:.2f}\n")
    f.write("\nGrouped Segments (raw detections):\n")
    for seg in segments:
        if seg[0] == seg[1]:
            f.write(f"Ball: {seg[0]:.1f}s\n")
        else:
            f.write(f"Ball: {seg[0]:.1f}s - {seg[1]:.1f}s\n")
print(f"Ball timestamps saved to {timestamp_file}")

# -----------------------------
# 5b. Save full detection annotated video (all frames processed) in 'full_detection'
# -----------------------------
full_detection_output = os.path.join("full_detection", "full_detection.mp4")
cap_full = cv2.VideoCapture(video_path)
writer_full = cv2.VideoWriter(full_detection_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
frame_idx = 0
while True:
    ret, frame = cap_full.read()
    if not ret:
        break
    ball_detected = False
    if frame_idx < len(results):
        result = results[frame_idx]
        if hasattr(result, 'boxes') and len(result.boxes) > 0:
            try:
                cls_ids = result.boxes.cls.cpu().numpy() if isinstance(result.boxes.cls, torch.Tensor) else result.boxes.cls
            except Exception as e:
                cls_ids = []
            if ball_class_id in cls_ids:
                ball_detected = True

    annotated_frame = frame.copy()
    # Overlay current time on top-right corner
    current_time = frame_idx / fps
    time_text = f"Time: {current_time:.2f}s"
    (text_width, text_height), _ = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.putText(annotated_frame, time_text, (width - text_width - 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # If ball is detected, overlay "Ball Detected" on top-left corner
    if ball_detected:
        cv2.putText(annotated_frame, "Ball Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    writer_full.write(annotated_frame)
    frame_idx += 1
cap_full.release()
writer_full.release()
print(f"Full detection video saved to {full_detection_output}")

# -----------------------------
# 5c. Save highlight segments (with and without annotations) in 'highlights'
# -----------------------------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# (Re)use the original cap by resetting its position
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
for i, (start, end) in enumerate(highlight_segments):
    output_without_tag = os.path.join("highlights", f"highlight_{i+1}_without_tag.mp4")
    output_with_tag = os.path.join("highlights", f"highlight_{i+1}_with_tag.mp4")
    
    writer_without = cv2.VideoWriter(output_without_tag, fourcc, fps, (width, height))
    writer_with = cv2.VideoWriter(output_with_tag, fourcc, fps, (width, height))
    
    print(f"Saving highlight {i+1}: from {start:.2f} to {end:.2f} seconds.")
    start_frame = int(start * fps)
    end_frame = int(end * fps)
    
    # Get the raw detection times for this segment (without buffer)
    detection_segment = segments[i]
    if detection_segment[0] == detection_segment[1]:
        detection_str = f"Ball: {detection_segment[0]:.1f}s"
    else:
        detection_str = f"Ball: {detection_segment[0]:.1f}s - {detection_segment[1]:.1f}s"
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Write the original frame to the "without tag" version
        writer_without.write(frame)
        
        # For the "with tag" version, annotate the frame
        annotated_frame = frame.copy()
        current_time = frame_num / fps
        time_text = f"Time: {current_time:.2f}s"
        (text_width, text_height), _ = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.putText(annotated_frame, time_text, (width - text_width - 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(annotated_frame, detection_str, (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        writer_with.write(annotated_frame)
    
    writer_without.release()
    writer_with.release()

cap.release()
print("All outputs saved:")
print("1. Full detection video (with annotations) in 'full_detection' folder.")
print("2. Ball timestamps text file in 'ball_timestamp' folder.")
print("3. Highlight segments (with and without annotations) in 'highlights' folder.")
