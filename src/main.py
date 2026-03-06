import time
import mediapipe as mp
import cv2
import numpy as np

from exercises import OneArmDumbellCurl
from core.tracker import RepTracker
from util import extract_landmarks, read_landmark, calculate_angle_3d

def webcamDemo():
    """
    Webcam'den canlı görüntü alarak poz tespiti yapar.
    Sol biceps açısını hesaplayıp konsola yazdırır.
    'q' tuşu ile çıkılır.
    """
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    video_capture = cv2.VideoCapture(0)
    
    exercise = OneArmDumbellCurl()
    tracker = RepTracker(start_phase=exercise.start_phase, min_rom=20.0)
    valid_reps = 0
    total_reps = 0
    
    with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
        
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            
            # recolor the cv2 image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            image.flags.writeable = False # saves memory while processing image
            result = pose.process(image)
            
            image.flags.writeable = True           

            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            landmarks = extract_landmarks(result)
            if landmarks:
                joints = [read_landmark(j, landmarks, mp_pose) for j in exercise.primary_joints]
                angle = calculate_angle_3d(joints)
                
                # Form analizi
                timestamp = time.time()
                rep_completed = tracker.add_frame(angle, timestamp)
                
                if rep_completed:
                    total_reps += 1
                    rep_data = tracker.extract_rep()
                    is_valid = exercise.apply_rules(rep_data)
                    
                    if is_valid:
                        valid_reps += 1
                        print(f"VALID REP! Valid: {valid_reps} / Total: {total_reps}")
                    else:
                        print(f"INVALID REP! Valid: {valid_reps} / Total: {total_reps}")

                # Ekrana yazdır
                cv2.putText(image, f"Angle: {int(angle)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(image, f"State: {tracker.state}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(image, f"Reps: {valid_reps}/{total_reps}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            else:
                pass

            # draw landmarks on image
            mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            cv2.imshow("Mediapipe Output",image)
            if(cv2.waitKey(10) & 0xFF == ord('q')):
                break
    video_capture.release()
    cv2.destroyAllWindows()

def main():
    webcamDemo()


if __name__ == "__main__":
    main()