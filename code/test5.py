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

# Open the video using OpenCV and get properties
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
# 5. Save each highlight segment as two versions:
#    - One WITHOUT annotation ("without tag")
#    - One WITH annotation ("with tag") that overlays:
#         * The current time (top-right)
#         * The ball detection times (bottom-left)
# -----------------------------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
for i, (start, end) in enumerate(highlight_segments):
    output_without_tag = f"highlight_{i+1}_without_tag.mp4"
    output_with_tag = f"highlight_{i+1}_with_tag.mp4"
    
    writer_without = cv2.VideoWriter(output_without_tag, fourcc, fps, (width, height))
    writer_with = cv2.VideoWriter(output_with_tag, fourcc, fps, (width, height))
    
    print(f"Saving highlight {i+1}: from {start:.2f} to {end:.2f} seconds.")
    start_frame = int(start * fps)
    end_frame = int(end * fps)
    
    # Get the detection times for this segment from the original "segments" (without buffer)
    detection_segment = segments[i]
    if detection_segment[0] == detection_segment[1]:
        detection_str = f"Ball: {detection_segment[0]:.1f}s"
    else:
        detection_str = f"Ball: {detection_segment[0]:.1f}s - {detection_segment[1]:.1f}s"
    
    # Set the video pointer to the start of the segment
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame_index = start_frame

    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Write the original frame to the "without tag" video
        writer_without.write(frame)
        
        # For the "with tag" version, make a copy to annotate
        annotated_frame = frame.copy()
        
        # Overlay the current time on the top-right corner.
        current_time = frame_num / fps
        time_text = f"Time: {current_time:.2f}s"
        # Get text size to compute the position so text is flush right.
        (text_width, text_height), _ = cv2.getTextSize(time_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        cv2.putText(annotated_frame, time_text, (width - text_width - 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Overlay the detection time summary on the bottom-left corner.
        cv2.putText(annotated_frame, detection_str, (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        writer_with.write(annotated_frame)
        current_frame_index += 1

    writer_without.release()
    writer_with.release()

cap.release()
print("All highlight segments saved (with and without annotations).")
