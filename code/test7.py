import os
import cv2
import torch
from ultralytics import YOLO

# =============================
# Configuration & Folder Setup
# =============================

# Set the video file path
video_path = "../assets/soccer1.mp4" #or basketball.mp4
# Derive a video name from the file name (without extension)
video_name = os.path.splitext(os.path.basename(video_path))[0]

# Base output directory (adjust "MVP1" to your project root if needed)
base_output = os.path.join("MVP1", "result", video_name)
# Define subfolders for each output type
full_det_dir = os.path.join(base_output, "full_detection_annotated")
highlights_dir = os.path.join(base_output, "highlights")
highlights_ts_dir = os.path.join(base_output, "highlights_with_ball_timestamp")

# Create the directories if they don't exist
for folder in [full_det_dir, highlights_dir, highlights_ts_dir]:
    os.makedirs(folder, exist_ok=True)

# =============================
# 1. Run YOLO Detection on Video
# =============================

# Load the YOLO model (COCO-pretrained YOLO11n)
model = YOLO("yolo11n.pt")
# Run inference on the entire video (this returns a list of results per frame)
results = model(video_path, save=True, show=True)

# Open video to get properties
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
video_duration = frame_count / fps
cap.release()

# =============================
# 2. Find Ball Detection Timestamps
# =============================

ball_class_name = "sports ball"
ball_class_id = None
for cid, name in model.names.items():
    if name.lower() == ball_class_name.lower():
        ball_class_id = cid
        break

if ball_class_id is None:
    print("The 'sports ball' class was not found in the model's classes!")
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
    exit()

# =============================
# 3. Group Detections into Continuous Segments
#    (Merge detections within 2 seconds)
# =============================

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

# =============================
# 4. Add a 3-second Buffer for Highlight Segments
# =============================

highlight_segments = []
for start, end in segments:
    highlight_start = max(0, start - 3)
    highlight_end = min(video_duration, end + 3)
    highlight_segments.append((highlight_start, highlight_end))

print("Highlight segments (with buffers):", highlight_segments)

# =============================
# 5a. Save Full Detection Annotated Video
#     (Full video with YOLO bounding boxes, labels, and current time overlay)
# =============================

# Open a new capture for full detection
cap_full = cv2.VideoCapture(video_path)
full_det_output = os.path.join(full_det_dir, f"{video_name}_full_detection.mp4")
writer_full = cv2.VideoWriter(full_det_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
frame_idx = 0

while True:
    ret, frame = cap_full.read()
    if not ret:
        break

    # Make a copy for annotation
    annotated_frame = frame.copy()

    # If YOLO results are available for this frame, draw bounding boxes.
    if frame_idx < len(results):
        result = results[frame_idx]
        if hasattr(result, 'boxes') and len(result.boxes) > 0:
            try:
                boxes = result.boxes.xyxy.cpu().numpy() if isinstance(result.boxes.xyxy, torch.Tensor) else result.boxes.xyxy
                confs = result.boxes.conf.cpu().numpy() if isinstance(result.boxes.conf, torch.Tensor) else result.boxes.conf
                cls_ids = result.boxes.cls.cpu().numpy() if isinstance(result.boxes.cls, torch.Tensor) else result.boxes.cls
            except Exception as e:
                boxes, confs, cls_ids = [], [], []
            for (x1, y1, x2, y2), conf, cls_id in zip(boxes, confs, cls_ids):
                label = model.names[int(cls_id)]
                # Draw bounding box
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # Put label with confidence above the box
                cv2.putText(annotated_frame, f"{label} {conf:.2f}", (int(x1), int(y1)-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Overlay the current time (top-right)
    current_time = frame_idx / fps
    time_text = f"Time: {current_time:.2f}s"
    (t_width, t_height), _ = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.putText(annotated_frame, time_text, (width - t_width - 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    writer_full.write(annotated_frame)
    frame_idx += 1

cap_full.release()
writer_full.release()
print(f"Full detection annotated video saved to {full_det_output}")

# =============================
# 5b. Save Highlight Segments (Without Annotation)
# =============================

# Re-open video capture for highlights
cap_high = cv2.VideoCapture(video_path)
for i, (start, end) in enumerate(highlight_segments):
    output_path = os.path.join(highlights_dir, f"highlight_{i+1}.mp4")
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    print(f"Saving highlight {i+1} (no annotation): from {start:.2f}s to {end:.2f}s.")
    start_frame = int(start * fps)
    end_frame = int(end * fps)
    
    cap_high.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_num in range(start_frame, end_frame):
        ret, frame = cap_high.read()
        if not ret:
            break
        writer.write(frame)
    writer.release()
cap_high.release()

# =============================
# 5c. Save Highlight Segments with Ball Timestamp Annotation
#      (Overlay current time at top-right and detection interval at bottom-left)
# =============================

cap_high_ts = cv2.VideoCapture(video_path)
for i, (start, end) in enumerate(highlight_segments):
    output_path = os.path.join(highlights_ts_dir, f"highlight_{i+1}_with_timestamp.mp4")
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    print(f"Saving highlight {i+1} with timestamp: from {start:.2f}s to {end:.2f}s.")
    start_frame = int(start * fps)
    end_frame = int(end * fps)
    
    # Get raw detection times (without buffer) for this segment from grouped segments
    detection_segment = segments[i]
    if detection_segment[0] == detection_segment[1]:
        detection_str = f"Ball: {detection_segment[0]:.1f}s"
    else:
        detection_str = f"Ball: {detection_segment[0]:.1f}s - {detection_segment[1]:.1f}s"
    
    cap_high_ts.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_num in range(start_frame, end_frame):
        ret, frame = cap_high_ts.read()
        if not ret:
            break
        
        annotated_frame = frame.copy()
        # Overlay current time (top-right)
        current_time = frame_num / fps
        time_text = f"Time: {current_time:.2f}s"
        (t_width, t_height), _ = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.putText(annotated_frame, time_text, (width - t_width - 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        # Overlay detection interval (bottom-left)
        cv2.putText(annotated_frame, detection_str, (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        writer.write(annotated_frame)
    writer.release()
cap_high_ts.release()

print("All outputs saved with the following structure:")
print(f"Video folder: {os.path.join('MVP1', 'result', video_name)}")
print(" - Full detection annotated video in 'full_detection_annotated'")
print(" - Highlight segments (without annotation) in 'highlights'")
print(" - Highlight segments with ball timestamp in 'highlights_with_ball_timestamp'")
