def extract_landmarks(results):
   """
   MediaPipe sonuçlarından landmark'ları çıkarır.
  
   Args:
       results: MediaPipe pose.process() çıktısı
  
   Returns:
       Landmark listesi veya None (tespit edilemezse)
   """
   try:
        landmarks = results.pose_world_landmarks.landmark
   except:
        return None
   return landmarks


def read_landmark(name, landmarks):
    """
   Belirtilen isimle landmark'ı okur.
  
   Args:
       name: Landmark ismi (örn: "LEFT_SHOULDER")
       landmarks: MediaPipe landmark listesi
  
   Returns:
       Landmark objesi (x, y, z koordinatları içerir)
   """
    return landmarks[mp_pose.PoseLandmark[name].value]