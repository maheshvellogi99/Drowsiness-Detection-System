import cv2
import numpy as np
import argparse
import imutils
import time
from imutils.video import VideoStream
from threading import Thread, Lock
import os
import face_recognition
import logging
from typing import Dict, Tuple, Optional
import platform
import urllib.request

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration parameters
CONFIG = {
    'EYE_AR_THRESH': 0.3,
    'EYE_AR_CONSEC_FRAMES': 30,
    'YAWN_THRESH': 20,
    'FRAME_WIDTH': 450,
    'FPS': 30,
    'ALARM_MESSAGES': {
        'drowsy': 'wake up sir',
        'yawn': 'Take some air sir'
    },
    'MOBILE_IP': '172.20.10.4',  # Your mobile device's IP address
    'MOBILE_PORT': '8080'  # Default port for IP Webcam
}

class DrowsinessDetector:
    def __init__(self, use_iphone: bool = False):
        self.killswitch_activated = False
        self.alarm_status = False
        self.alarm_status2 = False
        self.saying = False
        self.counter = 0
        self.alarm_lock = Lock()
        self.use_iphone = use_iphone
        self.vs = None
        self._initialize_camera()

    def _initialize_camera(self):
        try:
            logger.info("Initializing video stream...")
            
            if self.use_iphone:
                # Construct the URL for mobile camera stream
                mobile_url = f"http://{CONFIG['MOBILE_IP']}:{CONFIG['MOBILE_PORT']}/video"
                
                # Test connection to mobile camera
                try:
                    urllib.request.urlopen(mobile_url)
                    logger.info("Successfully connected to mobile camera")
                except Exception as e:
                    logger.error(f"Failed to connect to mobile camera: {str(e)}")
                    logger.info("Please ensure:")
                    logger.info("1. IP Webcam app is running on your mobile device")
                    logger.info("2. Your computer and mobile are on the same network")
                    logger.info("3. The IP address and port are correct")
                    logger.info(f"Current IP: {CONFIG['MOBILE_IP']}, Port: {CONFIG['MOBILE_PORT']}")
                    raise

                # Initialize video stream from mobile
                self.vs = VideoStream(src=mobile_url).start()
                time.sleep(1.0)  # Give time for the stream to initialize
                
                # Verify stream is working
                if self.vs.read() is None:
                    raise Exception("Failed to read from mobile camera stream")
                logger.info("Mobile camera stream initialized successfully")
            else:
                # Original webcam initialization code
                if platform.system() == 'Darwin':
                    for backend in [cv2.CAP_ANY, cv2.CAP_AVFOUNDATION]:
                        try:
                            self.vs = VideoStream(src=0, usePiCamera=False, 
                                                backend=backend).start()
                            time.sleep(1.0)
                            if self.vs.read() is not None:
                                logger.info(f"Camera initialized successfully with backend {backend}")
                                return
                        except Exception as e:
                            logger.warning(f"Failed to initialize with backend {backend}: {str(e)}")
                            continue
                    raise Exception("Failed to initialize camera with any backend")
                else:
                    self.vs = VideoStream(src=0).start()
                    time.sleep(1.0)
                    if self.vs.read() is not None:
                        logger.info("Camera initialized successfully")
                    else:
                        raise Exception("Failed to initialize camera")

        except Exception as e:
            logger.error(f"Failed to initialize camera: {str(e)}")
            raise

    def alarm(self, msg: str):
        with self.alarm_lock:
            while self.alarm_status:
                logger.info(f"Alarm triggered: {msg}")
                os.system(f'espeak "{msg}"')

            if self.alarm_status2:
                logger.info(f"Secondary alarm triggered: {msg}")
                self.saying = True
                os.system(f'espeak "{msg}"')
                self.saying = False

    @staticmethod
    def eye_aspect_ratio(eye: np.ndarray) -> float:
        # Convert eye landmarks to numpy array if they aren't already
        eye = np.array(eye)
        # Calculate the vertical distances
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        # Calculate the horizontal distance
        C = np.linalg.norm(eye[0] - eye[3])
        # Calculate the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear

    def final_ear(self, shape: Dict) -> Tuple[float, np.ndarray, np.ndarray]:
        # Convert eye landmarks to numpy arrays
        leftEye = np.array(shape['left_eye'])
        rightEye = np.array(shape['right_eye'])
        # Calculate EAR for both eyes
        leftEAR = self.eye_aspect_ratio(leftEye)
        rightEAR = self.eye_aspect_ratio(rightEye)
        return (leftEAR + rightEAR) / 2.0, leftEye, rightEye

    @staticmethod
    def lip_distance(shape: Dict) -> float:
        # Convert lip landmarks to numpy arrays
        top_lip = np.array(shape['top_lip'])
        low_lip = np.array(shape['bottom_lip'])
        # Calculate mean positions
        top_mean = np.mean(top_lip, axis=0)
        low_mean = np.mean(low_lip, axis=0)
        # Calculate vertical distance
        return abs(top_mean[1] - low_mean[1])

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = imutils.resize(frame, width=CONFIG['FRAME_WIDTH'])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = face_recognition.face_locations(gray)
        
        for (top, right, bottom, left) in rects:
            shape = face_recognition.face_landmarks(frame, [(top, right, bottom, left)])
            if not shape:
                continue

            shape = shape[0]
            ear, leftEye, rightEye = self.final_ear(shape)
            distance = self.lip_distance(shape)

            # Draw eye contours
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull, rightEyeHull], -1, (0, 255, 0), 1)

            # Draw lip contour
            lip = np.array(shape['top_lip'] + shape['bottom_lip'])
            cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

            # Check for drowsiness
            if ear < CONFIG['EYE_AR_THRESH']:
                self.counter += 1
                if self.counter >= CONFIG['EYE_AR_CONSEC_FRAMES']:
                    if not self.alarm_status:
                        self.alarm_status = True
                        Thread(target=self.alarm, args=(CONFIG['ALARM_MESSAGES']['drowsy'],), daemon=True).start()
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                self.counter = 0
                self.alarm_status = False

            # Check for yawning
            if distance > CONFIG['YAWN_THRESH']:
                cv2.putText(frame, "Yawn Alert", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not self.alarm_status2 and not self.saying:
                    self.alarm_status2 = True
                    Thread(target=self.alarm, args=(CONFIG['ALARM_MESSAGES']['yawn'],), daemon=True).start()
            else:
                self.alarm_status2 = False

            # Display metrics
            cv2.putText(frame, f"EYE: {ear:.2f}", (300, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"YAWN: {distance:.2f}", (300, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame

    def run(self):
        logger.info("Starting drowsiness detection...")
        frame_time = 1.0 / CONFIG['FPS']

        while not self.killswitch_activated:
            start_time = time.time()
            
            frame = self.vs.read()
            if frame is None:
                logger.error("Failed to grab frame")
                break

            frame = self.process_frame(frame)
            cv2.imshow("Drowsiness Detection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("k") or key == 27:  # 'k' or ESC to exit
                self.killswitch_activated = True
            elif key == ord("q"):
                break

            # Maintain FPS
            elapsed_time = time.time() - start_time
            if elapsed_time < frame_time:
                time.sleep(frame_time - elapsed_time)

        self.cleanup()

    def cleanup(self):
        logger.info("Cleaning up...")
        cv2.destroyAllWindows()
        if self.vs:
            self.vs.stop()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--iphone", action="store_true",
                    help="use iPhone camera instead of webcam")
    args = vars(ap.parse_args())

    try:
        detector = DrowsinessDetector(use_iphone=args["iphone"])
        detector.run()
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
