import cv2
import numpy as np
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

class FaceAnalyzer:
    """Advanced face analyzer focusing on concealment detection"""
    def __init__(self, config):
        self.config = config
        self.face_detector = self._load_face_detector()
        
    def _load_face_detector(self):
        """Load face detection model with error handling"""
        try:
            interpreter = tf.lite.Interpreter(model_path=self.config.FACE_MODEL_PATH)
            interpreter.allocate_tensors()
            return interpreter
        except Exception as e:
            logger.error(f"Error loading face detector: {str(e)}")
            return None
            
    def analyze_face(self, frame):
        """
        Enhanced face concealment detection
        
        Args:
            frame (numpy.ndarray): Input video frame
        
        Returns:
            tuple: (is_concealed, concealment_score)
        """
        try:
            # Detect faces
            faces = self._detect_faces(frame)
            
            # No faces detected or face detector failed
            if faces is None or len(faces) == 0:
                return True, 1.0  # Maximum concealment score
            
            # Analyze concealment for each detected face
            max_concealment_score = 0.0
            for face in faces:
                concealment_score = self._analyze_concealment(frame, face)
                max_concealment_score = max(max_concealment_score, concealment_score)
            
            # Determine if face is concealed based on threshold
            is_concealed = max_concealment_score > self.config.FACE_CONCEALMENT_THRESHOLD
            
            return is_concealed, max_concealment_score
            
        except Exception as e:
            logger.error(f"Face analysis error: {str(e)}")
            return True, 1.0  # Default to maximum concealment if error occurs
    
    def _detect_faces(self, frame):
        """Advanced face detection with multiple analysis techniques"""
        if self.face_detector is None:
            # Fallback to Haar cascade if TFLite model fails
            haar_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = haar_detector.detectMultiScale(gray, 1.1, 4)
            return faces
        
        # TFLite face detection logic
        input_details = self.face_detector.get_input_details()
        output_details = self.face_detector.get_output_details()
        
        # Preprocess frame
        preprocessed = cv2.resize(frame, (128, 128))
        preprocessed = preprocessed.astype(np.float32) / 255.0
        preprocessed = np.expand_dims(preprocessed, axis=0)
        
        # Run detection
        self.face_detector.set_tensor(input_details[0]['index'], preprocessed)
        self.face_detector.invoke()
        
        # Get detection results
        boxes = self.face_detector.get_tensor(output_details[0]['index'])[0]
        scores = self.face_detector.get_tensor(output_details[1]['index'])[0].flatten()
        
        # Apply sigmoid and threshold
        scores = 1 / (1 + np.exp(-np.clip(scores, -88.0, 88.0)))
        valid_detections = scores > self.config.FACE_DETECTION_THRESHOLD
        
        return boxes[valid_detections]
    
    def _analyze_concealment(self, frame, face_box):
        """Advanced concealment analysis"""
        x1, y1, x2, y2 = face_box[:4]
        face_region = frame[int(y1):int(y2), int(x1):int(x2)]
        
        if face_region.size == 0:
            return 1.0  # Maximum concealment if no region
        
        # Compute multiple concealment indicators
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Edge density
        edges = cv2.Canny(gray, 100, 200)
        edge_score = np.mean(edges > 0)
        
        # Texture variation
        texture_score = np.std(gray)
        
        # Color uniformity
        color_variance = np.var(face_region.reshape(-1, 3), axis=0).mean()
        
        # Combine scores (lower scores indicate more concealment)
        concealment_score = 1.0 - (
            0.4 * edge_score +  # Edge complexity
            0.3 * (texture_score / 255.0) +  # Texture variation
            0.3 * (1 - color_variance / 255.0)  # Color uniformity
        )
        
        return np.clip(concealment_score, 0.0, 1.0)