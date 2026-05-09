from detector import PadelDetector
from visualizer import PadelVisualizer
import cv2

import os
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_LOGGING_RULES"] = "*=false"

#Load and display video
cv2.namedWindow("Video",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video",800,600)

video_path = "/home/decimal/Downloads/input_sample_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Couldn't open video.")
    exit()

padel_detector = PadelDetector()
padel_visualizer = PadelVisualizer()
frame_count = 0
SKIP = 2

#Perfroms detections on every frame and draws bounding boxes around the detected objects and displays the video
while cap.isOpened():
    ret, frame = cap.read()
    frame_count += 1
    if not ret:
        break
    if frame_count % SKIP !=0:
        continue

    detections = padel_detector.detect(frame)
    padel_visualizer.draw_detections(frame,detections)
    cv2.rectangle(frame, (350, 30), (1450, 980), (255, 0, 0), 2)
    cv2.putText(frame, f"Players in court: {len(detections)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
    
    cv2.imshow("Video",frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()