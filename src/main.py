import mediapipe as mp
import cv2
import numpy as np

from maths import calculate_angle_3d
from util import extract_landmarks, read_landmark

def webcam_demo():
    """
    Webcam'den canlı görüntü alarak poz tespiti yapar.
    Sol biceps açısını hesaplayıp konsola yazdırır.
    'q' tuşu ile çıkılır.
    """
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    video_capture = cv2.VideoCapture(0)
    
    
    with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
        
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            
            # recolor the cv2 image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            image.flags.writeable = False # saves memory while processing image
            results = pose.process(image)
            image.flags.writeable = True           

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            landmarks = extract_landmarks(results)
            if landmarks:
                print("left biceps angle")
                print(calculate_angle_3d(read_landmark("LEFT_SHOULDER", landmarks),
                                         read_landmark("LEFT_ELBOW", landmarks),
                                         read_landmark("LEFT_WRIST", landmarks)))
            else:
                print("No landmarks detected")

            # draw landmarks on image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            cv2.imshow("Mediapipe Output",image)
            if(cv2.waitKey(10) & 0xFF == ord('q')):
                break
    video_capture.release()
    cv2.destroyAllWindows()

def main():
    webcam_demo()


if __name__ == "__main__":
    main()