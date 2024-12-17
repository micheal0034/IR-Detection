import logging
import numpy as np

logger = logging.getLogger(__name__)

class ImposterDetector:
    """Enhanced imposter detection with path analysis"""
    def __init__(self, thermal_analyzer, face_analyzer, movement_analyzer):
        self.thermal_analyzer = thermal_analyzer
        self.face_analyzer = face_analyzer
        self.movement_analyzer = movement_analyzer
        
        # Path analysis thresholds
        self.max_allowed_distance = 500  # Maximum allowed travel distance
        self.max_direction_changes = 5  # Maximum allowed direction changes
        self.linearity_threshold = 0.3

        self.camera_tracking = {}
    
    def detect_imposter(self, frame, detection_box, current_camera):
        """
        Detect potential imposter based on multi-modal analysis and cross-camera tracking
        
        Args:
            frame (numpy.ndarray): Current video frame
            detection_box (list): Bounding box of detected object
            current_camera (str): Current camera identifier
        
        Returns:
            tuple: (is_imposter, confidence_score)
        """
        try:
            # Analyze each modality
            is_thermal_anomalous, thermal_score = self.thermal_analyzer.analyze_thermal_signature(frame)
            is_face_concealed, face_score = self.face_analyzer.analyze_face(frame)
            
            # Movement and path analysis
            is_movement_anomalous, movement_score, path_info = self.movement_analyzer.analyze_movement(frame, detection_box)
            
            # Analyze path characteristics
            path_anomaly = self._analyze_path(path_info)
            
            # Cross-camera tracking
            cross_camera_anomaly = self._track_across_cameras(detection_box, current_camera)
            
            # Combine modality scores
            modality_scores = [
                thermal_score, 
                face_score, 
                movement_score
            ]
            
            # Calculate confidence score
            total_score = np.mean(modality_scores)
            
            # Determine if imposter based on multiple criteria
            is_imposter = (
                is_thermal_anomalous or 
                is_face_concealed or 
                is_movement_anomalous or 
                path_anomaly or 
                cross_camera_anomaly
            )
            
            return is_imposter, total_score
        
        except Exception as e:
            logger.error(f"Error in imposter detection: {str(e)}")
            return False, 0.0
    
    def _track_across_cameras(self, detection_box, current_camera):
        """
        Track object movement across different cameras
        
        Args:
            detection_box (list): Bounding box of detected object
            current_camera (str): Current camera identifier
        
        Returns:
            bool: True if suspicious cross-camera movement
        """
        # Get object center
        x1, y1, x2, y2 = detection_box[:4]
        center = ((x1 + x2) / 2, (y1 + y2) / 2)
        
        # Check if this object was seen in another camera recently
        if current_camera not in self.camera_tracking:
            self.camera_tracking[current_camera] = {}
        
        # Simple tracking logic
        for existing_camera, objects in self.camera_tracking.items():
            if existing_camera != current_camera:
                for obj_id, obj_data in objects.items():
                    # Check if this object has moved between cameras suspiciously
                    if self._calculate_distance(center, obj_data['center']) < 100:
                        logger.warning(f"Suspicious cross-camera movement detected")
                        return True
        
        # Update current camera tracking
        obj_id = len(self.camera_tracking[current_camera]) + 1
        self.camera_tracking[current_camera][obj_id] = {
            'center': center,
            'timestamp': logging.getLogger(__name__).manager.disable  # Current timestamp
        }
        
        return False
    
    def _calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _analyze_path(self, path_info):
        """Existing path analysis method remains the same"""
        if not path_info:
            return False
        
        # Check total distance traveled
        if path_info['total_distance'] > self.max_allowed_distance:
            logger.warning("Suspicious: Excessive travel distance")
            return True
        
        # Check direction changes
        if path_info['direction_changes'] > self.max_direction_changes:
            logger.warning("Suspicious: Frequent direction changes")
            return True
        
        # Check path linearity
        if not path_info.get('is_linear', True):
            logger.warning("Suspicious: Non-linear movement pattern")
            return True
        
        return False