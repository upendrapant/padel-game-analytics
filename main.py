import os
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_LOGGING_RULES"] = "*=false"


from detector import PadelDetector
from visualizer import PadelVisualizer
from pose_estimator import PadelPoseEstimator
import cv2

#Load and display video
cv2.namedWindow("Video",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video",800,600)

video_path = "/home/decimal/Downloads/input_sample_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Couldn't open video.")
    exit()

padel_detector = PadelDetector()
padel_pose_estimator = PadelPoseEstimator()
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
    detections = padel_pose_estimator.estimate_pose(frame, detections)
    padel_visualizer.draw_detections(frame,detections)  

    cv2.imshow("Video",frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
padel_pose_estimator.close()