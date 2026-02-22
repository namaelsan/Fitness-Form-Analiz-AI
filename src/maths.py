import numpy as np

def calculate_angle_3d(points: list):
    """
    Üç 3D eklem noktası arasındaki açıyı hesaplar.
  
    Args:
        a, b, c: MediaPipe landmark obje indexleri (LEFT_SHOULDER, LEFT_WRIST, LEFT_ELBOW...)
        b noktası açının köşesidir.
  
    Returns:
        Derece cinsinden açı (0-180 arası)
    """
    for i in range(3):
        points[i] = np.array([points[i].x, points[i].y, points[i].z])
    
    radians = np.arctan2(points[2][1]-points[1][1], points[2][0]-points[1][0]) - np.arctan2(points[0][1]-points[1][1], points[0][0]-points[1][0])
    angle = np.abs(radians*180/np.pi)
    
    if (angle > 180):
        angle = 360 - angle
    return angle