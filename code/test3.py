from ultralytics import YOLO
import torch

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Define the video file path
video_path = "/Users/dustinteng/Desktop/berkeley/Capstone/AI/videos/soccer1.mp4"

# Run inference on the video. The results object contains per-frame detections.
results = model(video_path, save=True, show=True)

# Find the class ID for "sports ball" from the model's names
ball_class_name = "sports ball"
ball_class_id = None

for class_id, name in model.names.items():
    if name.lower() == ball_class_name.lower():
        ball_class_id = class_id
        break

if ball_class_id is None:
    print("The 'sports ball' class was not found in the model's classes!")
    exit()

print(f"'Sports ball' class ID: {ball_class_id}")

# Set the video's frame rate (fps). Adjust if your video uses a different frame rate.
fps = 30  

# Iterate over the results for each frame
for frame_idx, result in enumerate(results):
    # Check if the frame has any detections
    if hasattr(result, 'boxes') and len(result.boxes) > 0:
        try:
            # Convert class detections to a numpy array if they are in a tensor format
            cls_ids = result.boxes.cls.cpu().numpy() if isinstance(result.boxes.cls, torch.Tensor) else result.boxes.cls
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            continue
        
        # Check if the ball (sports ball) is among the detected classes
        if ball_class_id in cls_ids:
            timestamp = frame_idx / fps
            print(f"Ball detected in frame {frame_idx} at {timestamp:.2f} seconds.")
