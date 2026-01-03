import mediapipe as mp
import cv2
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# will return the landmark with index name
def read_landmark(name, landmarks):
    return landmarks[mp_pose.PoseLandmark[name].value]

# will return landmarks or None
def extract_landmarks(results):
    try:
        landmarks = results.pose_world_landmarks.landmark
    except:
        return None
    return landmarks

# calculate the angle between 3 3d points
def calculate_angle_3d(a, b, c):
    """
    Calculates the angle between three 3D points a, b, c.
    a, b, c: MediaPipe landmark objects (with .x, .y, .z attributes).
    Returns: Angle in degrees.
    """
    # a_arr = np.array([a.x, a.y, a.z])
    # b_arr = np.array([b.x, b.y, b.z])
    # c_arr = np.array([c.x, c.y, c.z])
    
    # ba = a_arr - b_arr
    # bc = c_arr - b_arr
    
    # cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    
    # angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    # return np.degrees(angle)
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180/np.pi)
    
    if (angle > 180):
        angle = 360 - angle
    return angle
    
    
def webcam_demo():
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