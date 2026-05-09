import cv2

class PadelVisualizer:
    def __init__(self):
        pass

    def draw_detections(self,frame,detections):
        for det in detections:
            track_id = det["track_id"]
            x1, y1, x2, y2 = det["bbox"]
            conf = det["conf"]
            class_name = det["class"]
            label = f"{class_name} ID:{int(track_id)}" if track_id else class_name
            cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
            cv2.putText(frame,label,(int(x1),int(y1)-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
            
        return frame        