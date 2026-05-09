import cv2

class PadelVisualizer:
    def __init__(self):
        self.pose_connections = [
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5),
            (5, 6), (6, 8), (9, 10), (11, 12), (11, 13),
            (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
            (18, 20), (11, 23), (12, 24), (23, 24), (23, 25),
            (24, 26), (25, 27), (26, 28), (27, 29), (28, 30),
            (29, 31), (30, 32), (27, 31), (28, 32)
        ]

    def draw_detections(self,frame,detections):
        for det in detections:
            track_id = det.get("track_id")
            x1, y1, x2, y2 = det["bbox"]
            conf = det.get("conf", 0.0)
            class_name = det.get("class", "person")
            label = f"{class_name} ID:{int(track_id)}" if track_id is not None else class_name
            cv2.rectangle(frame,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
            cv2.putText(frame,label,(int(x1),int(y1)-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)

            #Court bounds and players count
            cv2.rectangle(frame, (200, 230), (1750, 950), (255, 0, 0), 2)
            cv2.putText(frame, f"Players in court: {len(detections)}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
            
            # Draw keypoints if available
            keypoints = det.get("keypoints")
            if keypoints:
                for connection in self.pose_connections:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    if start_idx < len(keypoints) and end_idx < len(keypoints):
                        kp1 = keypoints[start_idx]
                        kp2 = keypoints[end_idx]
                        if kp1["visibility"] > 0.5 and kp2["visibility"] > 0.5:
                            pt1 = (int(kp1["x"]), int(kp1["y"]))
                            pt2 = (int(kp2["x"]), int(kp2["y"]))
                            cv2.line(frame, pt1, pt2, (255, 0, 0), 2)
                for kp in keypoints:
                    if kp["visibility"] > 0.5:
                        pt = (int(kp["x"]), int(kp["y"]))
                        cv2.circle(frame, pt, 3, (0, 0, 255), -1)
        return frame        