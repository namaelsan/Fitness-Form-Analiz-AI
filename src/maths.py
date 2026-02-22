import numpy as np

# calculate the angle between 3 3d points
def calculate_angle_3d(a, b, c):
    """
    Üç 3D nokta arasındaki açıyı hesaplar.
  
    Args:
        a, b, c: MediaPipe landmark objeleri (.x, .y, .z özelliklerine sahip)
        b noktası açının köşesidir.
  
    Returns:
        Derece cinsinden açı (0-180 arası)
    """
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180/np.pi)
    
    if (angle > 180):
        angle = 360 - angle
    return angle