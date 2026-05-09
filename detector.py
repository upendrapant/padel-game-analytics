from ultralytics import YOLO

#Loads YOLOv8 model and performs object detection on a frame and returns detections list
class PadelDetector:
    def __init__(self, model_path="yolo11m.pt", tracker="bytetrack.yaml", conf=0.6, device='cuda',):
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device
        self.tracker = tracker
        self.roi = {"x_min": 200, "x_max": 1750, "y_min": 230, "y_max": 950}
    
    def _is_in_court(self, bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = y2 
        return (self.roi["x_min"] < cx < self.roi["x_max"] and
                self.roi["y_min"] < cy < self.roi["y_max"])

    def detect(self,frame):
        results = self.model.track(frame, persist=True, classes=[0], conf=self.conf, tracker=self.tracker, device=self.device, verbose=False)
        detections = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    track_id = box.id.item() if box.id is not None else None
                    if self._is_in_court([x1, y1, x2, y2]):
                        detections.append({"track_id": track_id, "bbox": [x1, y1, x2, y2], "class": "person", "conf": conf})
                        
        return detections