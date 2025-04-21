import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Initialize YOLOv8 model (replace with 'yolov8m.pt' for higher accuracy) 
model = YOLO('yolov8n.pt')

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30, embedder='mobilenet')

# Open video file or camera stream
cap = cv2.VideoCapture()

# Vehicle classes in COCO dataset (car, motorcycle, bus, truck)
vehicle_classes = [2, 3, 5, 7]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame, verbose=False)
    
    # Extract detections
    detections = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls)
            if cls_id in vehicle_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_id))

    # Update DeepSORT tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw tracked bounding boxes and IDs
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display output
    cv2.imshow('Vehicle Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()