from ultralytics import YOLO
import cv2
import time

# Load a COCO-pretrained YOLO model
model = YOLO("yolo11n.pt")

# Open video file
video_path = "/Users/dustinteng/Desktop/berkeley/Capstone/AI/videos/soccer1.mp4"
cap = cv2.VideoCapture(video_path)

# Get FPS and total frame count
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break  # End of video

    # Run YOLO inference on the frame
    results = model(frame)

    # Get timestamp (in seconds)
    timestamp = frame_count / fps
    frame_count += 1

    # Process detections
    person_count = 0
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])  # Get class ID
            if class_id == 0:  # COCO class '0' is 'person'
                person_count += 1

                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])  # Get confidence score
                
                # Customize bounding box color and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green Box
                cv2.putText(frame, f"Person {confidence:.2f}", (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Print timestamp and detected persons count
    print(f"Timestamp: {timestamp:.2f}s, Persons Detected: {person_count}")

    # Display the frame
    cv2.imshow("YOLO Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Press 'q' to exit

cap.release()
cv2.destroyAllWindows()
