import cv2
from ultralytics import YOLO

cv2.namedWindow("Video",cv2.WINDOW_NORMAL)
cv2.resizeWindow("Video",800,600)

video_path = "/home/decimal/Downloads/input_sample_video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Couldn't open video.")
    exit()

#Read and display frames in loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    cv2.imshow('Video',frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
