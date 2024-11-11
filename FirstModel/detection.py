import cv2
import torch
from ultralytics import YOLO

model = YOLO("Results/weights/best.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Couldn't open video stream.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        results = model.predict(frame, conf=0.60)  # Set confidence threshold to onyl display 60% above
        
        annotated_frame = results[0].plot()
        
        cv2.imshow('YOLO Detection', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()