# from security_system import SecuritySystem
# from utils import setup_logging

# def main():
#     # Setup logging
#     logger = setup_logging()

#     # Initialize the security system
#     security_system = SecuritySystem()

#     logger.info("Starting the security system...")
#     security_system.start_monitoring()

# if __name__ == "__main__":
#     main()



from security_system import SecuritySystem
from utils import setup_logging
import cv2

def main():
    # Setup logging
    logger = setup_logging()

    # Initialize the security system
    security_system = SecuritySystem()

    # Camera setup
    cameras = {
        'cam1': "rtsp://admin:amazingct123@192.168.0.160:554/Streaming/Channels/101",
        'cam2': "rtsp://admin:amazingct123@192.168.0.160:554/Streaming/Channels/201",
        # 'cam3': 2
    }

    # Open video captures for multiple cameras
    video_captures = {}
    for cam_name, cam_id in cameras.items():
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            logger.error(f"Could not open camera {cam_name}")
            continue
        video_captures[cam_name] = cap

    logger.info("Starting multi-camera monitoring...")

    try:
        while True:
            # Capture frames from all cameras
            camera_frames = {}
            for cam_name, cap in video_captures.items():
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Failed to capture frame from {cam_name}")
                    continue
                camera_frames[cam_name] = frame

            # Process frames
            security_system.process_multi_camera_frames(camera_frames)

            # Optional: Display frames
            for cam_name, frame in camera_frames.items():
                cv2.imshow(cam_name, frame)

            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        logger.error(f"Error in multi-camera monitoring: {str(e)}")

    finally:
        # Release resources
        for cap in video_captures.values():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()