# cred: https://github.com/ItayCoifman/blazepose-mediapipe-example/blob/c452b58c29b45c478f49efa2c462791ec6eaa226/tools/video.py

import IPython
from IPython.display import display, Javascript, Image,HTML
import os
import subprocess
import cv2
import numpy as np

class Video():
    """
    The Video class provides a convenient way to interact with video files. It allows you to create a Video object from a file path and provides information about the video's properties (e.g. fps, number of frames, width, and height).

    Attributes:
    path (str): The file path of the video.
    fps (int): The frames per second of the video.
    nFrames (int): The number of frames in the video.
    width (int): The width of the video frames in pixels.
    height (int): The height of the video frames in pixels.
    name (str): The base name of the video file.

    Methods:
    from_path(video_path): Creates a Video object from the given video path.
    play(frac=1): Displays the video with a given fraction of the original height and width.
    str: Returns a string representation of the video, including its name, width, height, number of frames, fps, and path.

    Examples:
    >>> video = Video.from_path('path/to/video.mp4')
    >>> print(video)
    Name- video.mp4:
    Width- 640, Height- 480
    number of frames- 1000, Fps- 30
    Path- path/to/video.mp4

    >>> video.play()
    # Displays the video in its original size
    """
    def __init__(self,path, fps,nFrames, width,height):
      self.path = path
      self.fps = fps
      self.nFrames = int(nFrames)
      self.width = width
      self.height = height
      self.name = os.path.basename(path)

    @classmethod
    def from_path(cls, video_path):
      vidcap = cv2.VideoCapture(video_path)
      fps = vidcap.get(cv2.CAP_PROP_FPS)
      nFrames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
      width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
      height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
      return cls(path = video_path,fps=fps,nFrames=nFrames, width=width,height=height)

    def crop(self, output_path, xyxy=None, start_frame=None, end_frame=None):
        """
        Creates a cropped version of the video using the given bounding box (xyxy) and frame range (start_frame, end_frame),
        and saves it to the specified output_path.
        
        Parameters:
        - output_path (str): The path where the cropped video will be saved.
        - xyxy (tuple of int, optional): A bounding box represented as (x1, y1, x2, y2), where (x1, y1) is the top-left 
            corner and (x2, y2) is the bottom-right corner of the box. If not provided, the entire video frame will be used.
        - start_frame (int, optional): The first frame to include in the cropped video. If not provided, defaults to 0.
        - end_frame (int, optional): The last frame to include in the cropped video. If not provided, defaults to the last frame
            in the original video.
        """
        # Open the input video file for reading
        cap = cv2.VideoCapture(self.path)
        
        # Determine the video properties
        fps = self.fps
        width = self.width
        height = self.height
        num_frames = self.nFrames
        
        # Determine the frame range to use
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            end_frame = num_frames - 1
        start_frame = max(start_frame, 0)
        end_frame = min(end_frame, num_frames - 1)
        num_frames = end_frame - start_frame + 1
        
        # Determine the region of interest (ROI) to crop
        if xyxy is None:
            roi = np.array([0, 0, width, height])
        else:
            roi = np.array(xyxy)
            roi[0::2] = np.clip(roi[0::2], 0, width)
            roi[1::2] = np.clip(roi[1::2], 0, height)
        roi_width = roi[2] - roi[0]
        roi_height = roi[3] - roi[1]
        
        # Open the output video file for writing
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (roi_width, roi_height))
        
        # Loop over the frames in the specified range
        for frame_idx in range(start_frame, end_frame + 1):
            # Read the next frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
                
            # Crop the frame to the ROI
            cropped_frame = frame[roi[1]:roi[3], roi[0]:roi[2], :]
            
            # Write the cropped frame to the output file
            out.write(cropped_frame)
        
        # Release the video capture and writer objects
        cap.release()
        out.release()

        
        # Create a new Video object for the cropped video
        cropped_video = Video.from_path(output_path)

                
        # Convert the output video to a more widely supported codec
        cropped_video.convert()
        
        return cropped_video

    def convert(self):
      dir_path = os.path.dirname(self.path) + '/'
      temp_path = dir_path + "tmp_" + self.name
      os.rename(self.path,temp_path)
      output_path = self.Convert_video(temp_path, self.path,delete_input=True)
    @staticmethod
    def Convert_video(input_path, output_path, delete_input=True):
      """
      Convert the input video to a more widely supported codec.

      Parameters:
      - input_path (str): The path to the input video file.
      - output_path (str): The path to the output video file.
      - delete_input (bool, optional): If set to True, the input video will be deleted after conversion. Defaults to True.

      Returns:
      - str: The path to the output video file if the conversion was successful, the path to the input video file otherwise.

      """
      command = [
          "ffmpeg",
          "-y",
          "-i",
          os.path.abspath(input_path),
          "-crf",
          "18",
          "-preset",
          "veryfast",
          "-hide_banner",
          "-loglevel",
          "error",
          "-vcodec",
          "libx264",
          output_path
      ]
      result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      if result.returncode == 0:
          print("Converted to colab freiendly format")
          if delete_input:
              os.remove(input_path)
          return output_path
      else:
          print("Command failed with return code {}.".format(result.returncode))
          print("Standard output:")
          print(result.stdout.decode())
          print("Standard error:")
          print(result.stderr.decode())
          return input_path

    def __str__(self):
      return f"Name- {self.name}:\n Width- {self.width}, Height- {self.height} \n number of frames- {self.nFrames}, Fps- {self.fps} \n Path-{self.path}"

    
    def play(self,frac = 1):
      display(
        IPython.display.Video(data=self.path, embed=True, height=int(self.height * frac), width=int(self.width * frac))
            )