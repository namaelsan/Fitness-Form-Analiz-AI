from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np

@dataclass
class LandmarkPoint:
    x: float
    y: float
    z: float

class PoseModel(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        pass
        
    @abstractmethod
    def process_image(self, image: np.ndarray) -> Any:
        """Processes the image and returns raw results object. image is RGB."""
        pass
        
    @abstractmethod
    def extract_landmarks(self, results: Any) -> Optional[Dict[str, LandmarkPoint]]:
        """Extracts landmarks into a standardized dictionary keyed by joint name."""
        pass
        
    @abstractmethod
    def draw_landmarks(self, image: np.ndarray, results: Any):
        """Draw landmarks over the given BGR image in-place."""
        pass
        
    def release(self):
        """Release resources if necessary."""
        pass

class MediaPipeModel(PoseModel):
    def __init__(self, complexity=1):
        import mediapipe as mp
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            model_complexity=complexity,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Verify that the model is actually functional (catches _graph is None errors)
        try:
            dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
            self.pose.process(dummy_img)
        except Exception as e:
            self.release()
            raise RuntimeError(f"MediaPipe Pose initialization failed for complexity {complexity}: {e}")
        
    def process_image(self, image: np.ndarray) -> Any:
        if not hasattr(self, 'pose') or self.pose is None:
            raise RuntimeError("MediaPipe Pose model is not initialized or has been released.")
        return self.pose.process(image)
        
    def extract_landmarks(self, results: Any) -> Optional[Dict[str, LandmarkPoint]]:
        if results is None or not hasattr(results, 'pose_world_landmarks') or not results.pose_world_landmarks:
            return None
            
        landmarks_dict = {}
        for landmark_name in self.mp_pose.PoseLandmark:
            idx = landmark_name.value
            lm = results.pose_world_landmarks.landmark[idx]
            landmarks_dict[landmark_name.name] = LandmarkPoint(x=lm.x, y=lm.y, z=lm.z)
            
        return landmarks_dict
        
    def draw_landmarks(self, image: np.ndarray, results: Any):
        if results is not None and hasattr(results, 'pose_landmarks') and results.pose_landmarks:
            self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
    def release(self):
        if hasattr(self, 'pose') and self.pose is not None:
            try:
                self.pose.close()
            except:
                pass
            self.pose = None

class YOLOv8Model(PoseModel):
    def __init__(self, model_version="yolov8n-pose.pt"):
        from ultralytics import YOLO
        self.model = YOLO(model_version)
        # YOLOv8 pose keypoint mapping to MediaPipe string names (approximate COCO to BLazePose)
        self.keypoint_mapping = {
            0: "NOSE", 1: "LEFT_EYE", 2: "RIGHT_EYE", 3: "LEFT_EAR", 4: "RIGHT_EAR",
            5: "LEFT_SHOULDER", 6: "RIGHT_SHOULDER", 7: "LEFT_ELBOW", 8: "RIGHT_ELBOW",
            9: "LEFT_WRIST", 10: "RIGHT_WRIST", 11: "LEFT_HIP", 12: "RIGHT_HIP",
            13: "LEFT_KNEE", 14: "RIGHT_KNEE", 15: "LEFT_ANKLE", 16: "RIGHT_ANKLE"
        }
        
    def process_image(self, image: np.ndarray) -> Any:
        # ultralytics expects BGR natively, but handles RGB. Since we pass RGB, usually we can just pass it directly.
        results = self.model(image, conf=0.6, verbose=False)
        return results[0]
        
    def extract_landmarks(self, results: Any) -> Optional[Dict[str, LandmarkPoint]]:
        if results.keypoints is None or len(results.keypoints) == 0:
            return None
        
        # Take first detected person
        # keypoints.data shape is usually (num_people, num_joints, 3) where 3 is x, y, confidence
        kp_data = results.keypoints.data[0].cpu().numpy()
        
        landmarks_dict = {}
        for idx, name in self.keypoint_mapping.items():
            if idx < len(kp_data):
                x, y, conf = kp_data[idx]
                if conf > 0.6:
                    # YOLO outputs image coordinates, we'll rough approximate z to 0 since YOLO is primarily 2D
                    # MediaPipe outputs normalized world coords. For angle tracking 2D x,y might suffice for lateral angles.
                    landmarks_dict[name] = LandmarkPoint(x=float(x), y=float(y), z=0.0)
        
        if not landmarks_dict:
            return None
            
        return landmarks_dict
        
    def draw_landmarks(self, image: np.ndarray, results: Any):
        # results.plot() returns a BGR image, we need to convert it to RGB
        import cv2
        annotated_frame = results.plot()
        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        # copy annotated back into image
        np.copyto(image, annotated_frame_rgb)

