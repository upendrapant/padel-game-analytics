import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class PadelPoseEstimator:
    def __init__(self, model_asset_path='pose_landmarker.task'):
        base_options = python.BaseOptions(model_asset_path=model_asset_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            output_segmentation_masks=False)
        self.detector = vision.PoseLandmarker.create_from_options(options)

    def estimate_pose(self, frame, detections):
        h, w = frame.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = map(int, det["bbox"])
            
            # Ensure coordinates are within frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Check for valid crop
            if x2 <= x1 or y2 <= y1:
                det["keypoints"] = None
                continue
                
            # Crop the bounding box for the person
            person_crop = frame[y1:y2, x1:x2]
            
            # Convert to RGB and then to mp.Image
            person_crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=person_crop_rgb)
            
            detection_result = self.detector.detect(mp_image)
            
            if len(detection_result.pose_landmarks) > 0:
                keypoints = []
                crop_h, crop_w = person_crop.shape[:2]
                
                # Map normalized keypoints back to original frame coordinates
                for landmark in detection_result.pose_landmarks[0]:
                    abs_x = x1 + (landmark.x * crop_w)
                    abs_y = y1 + (landmark.y * crop_h)
                    keypoints.append({
                        "x": abs_x,
                        "y": abs_y,
                        "z": landmark.z,
                        "visibility": landmark.presence if hasattr(landmark, 'presence') else landmark.visibility
                    })
                det["keypoints"] = keypoints
            else:
                det["keypoints"] = None

        return detections

    def close(self):
        if hasattr(self, 'detector') and self.detector is not None:
            self.detector.close()
