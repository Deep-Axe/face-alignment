import warnings
import cv2
import mediapipe as mp
import numpy as np
from ..core import FaceDetector

class MediaPipeDetector(FaceDetector):
    def __init__(self, device, min_detection_confidence=0.5, max_num_faces=1, verbose=False):
        super().__init__(device, verbose)
        warnings.warn('Warning: MediaPipeDetector is experimental. Please validate results for your use case.')
        
        # Use Face Mesh instead of Face Detection for 478 landmarks
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=max_num_faces,
            refine_landmarks=True,  # This enables 478 landmarks instead of 468
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        self.max_num_faces = max_num_faces

    def detect_from_image(self, tensor_or_path):
        image = self.tensor_or_path_to_ndarray(tensor_or_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        detected_faces = []
        if results.multi_face_landmarks:
            h, w, _ = image.shape
            for face_landmarks in results.multi_face_landmarks[:self.max_num_faces]:
                # Get bounding box from landmarks
                landmarks = np.array([[lm.x * w, lm.y * h] for lm in face_landmarks.landmark])
                x_min, y_min = np.min(landmarks, axis=0)
                x_max, y_max = np.max(landmarks, axis=0)
                
                # Add some padding
                padding = 20
                x1 = max(0, int(x_min - padding))
                y1 = max(0, int(y_min - padding))
                x2 = min(w, int(x_max + padding))
                y2 = min(h, int(y_max + padding))
                
                detected_faces.append([x1, y1, x2, y2])
        
        return detected_faces

    def get_landmarks_478(self, tensor_or_path):
        """Get 478 face landmarks directly"""
        image = self.tensor_or_path_to_ndarray(tensor_or_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        all_landmarks = []
        if results.multi_face_landmarks:
            h, w, _ = image.shape
            for face_landmarks in results.multi_face_landmarks[:self.max_num_faces]:
                # Convert to numpy array with x, y, z coordinates
                landmarks = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in face_landmarks.landmark])
                all_landmarks.append(landmarks)
        
        return all_landmarks

    @property
    def reference_scale(self):
        return 195

    @property
    def reference_x_shift(self):
        return 0

    @property
    def reference_y_shift(self):
        return 0