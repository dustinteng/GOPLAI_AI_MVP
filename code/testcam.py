from ultralytics import YOLO
import cv2

#test camera direct to computer


# Load a COCO-pretrained YOLO model
model = YOLO("yolo11n.pt")  # Change to your model if needed

# Open the default webcam (0) or change to another camera index if needed
cap = cv2.VideoCapture(0)

# Get FPS for timestamp calculations
fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30  # Default to 30 if unknown
frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break  # Stop if no frame is captured

    # Run YOLO inference on the frame
    results = model(frame)

    # Calculate timestamp (in seconds)
    timestamp = frame_count / fps
    frame_count += 1

    # Count detected people
    person_count = 0

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])  # Get class ID
            if class_id == 0:  # COCO class '0' is 'person'
                person_count += 1

                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])  

                # Calculate bottom center of the bounding box
                center_x = (x1 + x2) // 2
                bottom_y = y2

                # Draw a circle at the bottom center of detected persons
                cv2.circle(frame, (center_x, bottom_y), 10, (0, 255, 0), -1)  # Green filled circle

                # Display confidence score above the circle
                confidence = float(box.conf[0])
                cv2.putText(frame, f"{confidence:.2f}", (center_x - 15, bottom_y - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Print timestamp and number of people detected
    print(f"Timestamp: {timestamp:.2f}s, Persons Detected: {person_count}")

    # Show the output in a window
    cv2.imshow("Live YOLO Detection", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
