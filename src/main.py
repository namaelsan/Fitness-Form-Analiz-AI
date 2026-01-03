import mediapipe as mp
import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm

from classes import Video
from src.lib import Landmarks

def estimate(video: Video, output_path: str):
	output = []
	timestamps = []
	videoCapture = cv2.VideoCapture(video.path)        
	videoWriter = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), video.fps, (video.width, video.height))

	# Initialize the pose estimation model
	with mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
		for frame_index in tqdm(range(vid.nFrames)):
			success, img = vid_cap.read()
			if not success:
				break
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Get the timestamp of the current frame
			timestamps.append(vid_cap.get(cv2.CAP_PROP_POS_MSEC))
			results = pose.process(img)
			output.append(results)

            # Draw the pose annotation on the image.
			img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
			mp.solutions.drawing_utils.draw_landmarks(
                img,
                results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
                )
			# Write the frame to the output video
			videoWriter.write(img)
	videoWriter.release()
	cv2.destroyAllWindows()

	marker_df,visibility_df = Landmarks.landmarks_2_table(output,time_vec = np.array(timestamps)/1000)
 
	return Video.from_path(output_path), marker_df, visibility_df


def video_pose_estimation(video_path: str, output_path = None, verbose=True):
    """
    Apply a pose estimation model to each frame of a video and return a processed video and the output from the model for each frame.

    Args:
    video_path (str): File path of the video to be processed.
    output_path (str, optional): File path of the output video. Default is None.
    verbose (bool, optional): Boolean indicating whether to print progress updates. Default is True.

    Returns:
    vid (Video): Processed video object.
    marker_df (DataFrame): DataFrame containing the x, y, and z coordinates of each landmark for each frame of the processed video.
    visibility_df (DataFrame): DataFrame containing the visibility score for each landmark for each frame of the processed video.
    """
    output = []
    timestamps =[]
    # Todo add option to input video:
    vid = Video.from_path(video_path)
    video_name = os.path.basename(video_path)
    #init video writer and reader
    vid_cap = cv2.VideoCapture(vid.path) 
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), vid.fps, (vid.width, vid.height))

    # Initialize the pose estimation model
    with mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        for frame_index in tqdm(range(vid.nFrames)):
            success, img = vid_cap.read()
            if not success:
                break
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Get the timestamp of the current frame
            timestamps.append(vid_cap.get(cv2.CAP_PROP_POS_MSEC))
            # Run the model
            results = pose.process(img)
            output.append(results)
            # Draw the pose annotation on the image.
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            mp.solutions.drawing_utils.draw_landmarks(
                img,
                results.pose_landmarks,
                mp.solutions.pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style()
                )
                    # Write the frame to the output video
            video_writer.write(img)
    # close video
    video_writer.release()
    cv2.destroyAllWindows()
    # Return the processed video and the model outputs
    try:
      marker_df,visibility_df = landmarks_2_table(output,time_vec = np.array(timestamps)/1000)
    except:
      marker_df,visibility_df = output,[]
      print("plese replace the video")
    return Video.from_path(output_path), marker_df, visibility_df





from google.colab import files
uploaded = files.upload()
for file_name in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=file_name, length=len(uploaded[file_name])))
     


output_path = "proc.mp4"
vid, marker_df,visibility_df  = video_pose_estimation(video_path = file_name,output_path =output_path)
     


vid.convert()
vid.play(frac = 0.75) #change frac acording to the video width and height
     
marker_df.head()


