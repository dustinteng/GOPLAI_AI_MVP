import cv2
import numpy as np
import os
from ultralytics import YOLO

def overlay_transparent(background, overlay, x, y, overlay_size=None):
    """
    Overlay `overlay` onto `background` at position (x, y) with optional resizing.
    If the overlay image doesn't have an alpha channel, it is converted to BGRA.
    """
    bg = background.copy()
    if overlay_size is not None:
        overlay = cv2.resize(overlay, overlay_size, interpolation=cv2.INTER_AREA)
    
    # If overlay doesn't have alpha channel, add one.
    if overlay.shape[2] == 3:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2BGRA)
    
    b, g, r, a = cv2.split(overlay)
    overlay_rgb = cv2.merge((b, g, r))
    
    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha_mask = a.astype(float) / 255.0
    h, w, _ = overlay_rgb.shape

    # Get region of interest on background
    roi = bg[y:y+h, x:x+w]
    if roi.shape[0] != h or roi.shape[1] != w:
        # If the ROI goes outside of the background, adjust the overlay size.
        h = roi.shape[0]
        w = roi.shape[1]
        overlay_rgb = overlay_rgb[0:h, 0:w]
        alpha_mask = alpha_mask[0:h, 0:w]

    # Blend the overlay with the ROI
    for c in range(0, 3):
        roi[:, :, c] = (alpha_mask * overlay_rgb[:, :, c] + (1 - alpha_mask) * roi[:, :, c])
    
    bg[y:y+h, x:x+w] = roi
    return bg

# -----------------------------
# Configuration
# -----------------------------
video_path = "../assets/crown.MP4"
crown_image_path = "../assets/crown2.png"  # crown image with transparency (BGRA)

# Crown multiplier to adjust the crown size:
# Set crown_multiplier < 1 to make the crown smaller, > 1 to make it bigger.
crown_multiplier = 0.5  # adjust as needed

# -----------------------------
# Load crown image (with alpha channel)
# -----------------------------
crown_img = cv2.imread(crown_image_path, cv2.IMREAD_UNCHANGED)
if crown_img is None:
    raise FileNotFoundError("Could not load crown image. Make sure crown2.png exists.")

# -----------------------------
# Load YOLO model (COCO-pretrained YOLO11n)
# -----------------------------
model = YOLO("yolo11n.pt")

# Run inference on the video (this returns a list of results, one per frame)
results = model(video_path, save=True, show=True)

# -----------------------------
# Open video to read frames and get properties
# -----------------------------
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# -----------------------------
# Determine unique output file name
# -----------------------------
output_video_path = "../videos/soccer1_with_crown.mp4"
base_name, ext = os.path.splitext(output_video_path)
counter = 1
while os.path.exists(output_video_path):
    output_video_path = f"{base_name}_{counter}{ext}"
    counter += 1

# Prepare video writer for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# The COCO person class is usually id 0.
person_class_id = 0

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Check if there is a YOLO result for this frame
    if frame_idx < len(results):
        result = results[frame_idx]
        if hasattr(result, 'boxes') and len(result.boxes) > 0:
            try:
                # Get class IDs and bounding boxes for this frame
                cls_ids = result.boxes.cls.cpu().numpy() if hasattr(result.boxes.cls, 'cpu') else result.boxes.cls
                boxes = result.boxes.xyxy.cpu().numpy() if hasattr(result.boxes.xyxy, 'cpu') else result.boxes.xyxy
            except Exception as e:
                print(f"Error processing frame {frame_idx}: {e}")
                cls_ids = []
                boxes = []
            
            # Process each detected person
            for cls_id, box in zip(cls_ids, boxes):
                if int(cls_id) == person_class_id:
                    # box: [x1, y1, x2, y2]
                    x1, y1, x2, y2 = map(int, box)
                    # Calculate the top center of the bounding box
                    center_x = (x1 + x2) // 2
                    head_y = int(y1 + (abs(y1 - y2) / 10))
                    # Determine crown size.
                    # For example, crown width can be ~ 1.2x the width of the person times the multiplier.
                    person_width = x2 - x1
                    crown_width = int(0.5 * person_width * crown_multiplier)
                    # Maintain crown aspect ratio
                    aspect_ratio = crown_img.shape[0] / crown_img.shape[1]
                    crown_height = int(crown_width * aspect_ratio)
                    
                    # Determine where to place the crown.
                    # We want the crown's bottom center to be at (center_x, head_y)
                    crown_x = center_x - crown_width // 2
                    crown_y = head_y - crown_height  # Place it above the head
                    
                    # Check boundaries
                    crown_x = max(0, int(crown_x))
                    crown_y = max(0, int(crown_y))

                    # Overlay the crown image onto the frame
                    frame = overlay_transparent(frame, crown_img, crown_x, crown_y, overlay_size=(crown_width, crown_height))

    writer.write(frame)
    frame_idx += 1

cap.release()
writer.release()
print(f"Output video with crown overlay saved to {output_video_path}")
