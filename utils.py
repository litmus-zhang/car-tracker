import cv2
import numpy as np

def load_model():
    """Load MobileNet-SSD model"""
    prototxt = "models/deploy.prototxt"
    caffemodel = "models/mobilenet_iter_73000.caffemodel"
    return cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

def detect_vehicles(frame, net, confidence_threshold=0.5):
    """Process frame and return vehicle detections"""
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)
    net.setInput(blob)
    detections = net.forward()
    
    vehicles = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            class_id = int(detections[0, 0, i, 1])
            if class_id in [7, 8, 9]:  # Car, truck, bus
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                vehicles.append((x1, y1, x2, y2))
    
    return vehicles