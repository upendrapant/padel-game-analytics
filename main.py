import os
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["QT_QPA_PLATFORM"] = "xcb"
os.environ["QT_LOGGING_RULES"] = "*=false"


from detector import PadelDetector
from tracknet_tracker import TrackNetTracker
from visualizer import PadelVisualizer
from pose_estimator import PadelPoseEstimator
from shot_classifier import ShotClassifier
from analytics import ShotLogger, Analytics
import cv2

#Load and display video
cv2.namedWindow("Video",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video",1280,720)

video_path = "/home/decimal/Downloads/input_sample_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Couldn't open video.")
    exit()

padel_detector = PadelDetector()
padel_ball_tracker = TrackNetTracker(model_path="model_best.pt")
padel_pose_estimator = PadelPoseEstimator()
padel_visualizer = PadelVisualizer()
padel_shot_classifier = ShotClassifier()
logger = ShotLogger()

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0 or fps != fps:
    fps = 30.0

frame_num = 0

#Perfroms detections on every frame and draws bounding boxes around the detected objects and displays the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    detections = padel_detector.detect(frame)
    detections = padel_pose_estimator.estimate_pose(frame, detections)
    ball_coords = padel_ball_tracker.track_ball(frame)
    
    # pyrefly: ignore [bad-argument-count]
    shot_events = padel_shot_classifier.classify_shots(detections, ball_coords, frame_num, fps)
    logger.log(shot_events)
    
    padel_visualizer.draw_detections(frame, detections, ball_coords)  

    cv2.imshow("Video",frame)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        break
        
    frame_num += 1

cap.release()
cv2.destroyAllWindows()
padel_pose_estimator.close()

logger.save("output/")
analytics = Analytics(logger.shots)
analytics.print_summary()