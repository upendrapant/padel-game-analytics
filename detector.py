from ultralytics import YOLO

#Loads YOLOv8 model and performs object detection on a frame and returns detections list
class PadelDetector:
    def __init__(self, model_path="yolo11n.pt", conf=0.5, device='cuda',):
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = device

    def detect(self,frame):
        results = self.model.track(frame, persist=True, classes=[0], conf=self.conf, device=self.device, verbose=False)
        detections = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    track_id = box.id.item() if box.id is not None else None
                    detections.append({"track_id": track_id, "bbox": [x1, y1, x2, y2], "class": "person", "conf": conf})
        return detections