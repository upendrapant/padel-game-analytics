import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_LOGGING_RULES"] = "*=false"

import cv2
from ultralytics import YOLO

#Loads YOLOv8 model and performs object detection on a frame and returns detections list
class PadelDetector:
    def __init__(self, model_path="yolo11n.pt", conf=0.5, device='cuda',):
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device

    def detect(self,frame):
        results = self.model.track(frame, persist=True, classes=[0], conf=0.5,verbose=False)
        detections = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    track_id = box.id.item() if box.id is not None else None
                    detections.append({"track_id": track_id, "bbox": [x1, y1, x2, y2], "class": "person", "conf": conf})
        return detections

#Load and display video
cv2.namedWindow("Video",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video",800,600)

video_path = "/home/decimal/Downloads/input_sample_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Couldn't open video.")
    exit()

padel_detector = PadelDetector()

#Perfroms detections on every frame and draws bounding boxes around the detected objects and displays the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    detections = padel_detector.detect(frame)

    for det in detections:
        track_id = det["track_id"]
        x1, y1, x2, y2 = det["bbox"]
        conf = det["conf"]
        class_name = det["class"]

        cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
        cv2.putText(frame,f"{class_name} {track_id}",(int(x1),int(y1)-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)


    cv2.imshow('Video',frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
