import numpy as np
import pandas as pd

landmark_names = [
    'nose',
    'left_eye_inner', 'left_eye', 'left_eye_outer',
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear',
    'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_pinky', 'right_pinky',
    'left_index', 'right_index',
    'left_thumb', 'right_thumb',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
    'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index',
]


def landmark_2_array(landmarks):
    """
      Convert a list of landmark objects to a 3D numpy array of shape
      (time_steps, key_points, 4).

      Args:
      - landmarks: A list of landmark objects.

      Returns:
      - markers_array: A 3D numpy array containing the x, y, z coordinates
      and visibility of each key point for each frame.
      - visibility_array: A 2D numpy array containing the visibility of
      each key point for each frame.
    """
    time_steps = len(landmarks)
    key_points = len(landmarks[0].pose_world_landmarks.landmark)
    output = np.zeros((time_steps, key_points, 4))
    for i in range(time_steps):
        world_landmarks_i = landmarks[i].pose_world_landmarks.landmark
        for j in range(key_points):
            world_landmarks_i_j = world_landmarks_i[j]
            output[i, j, 2] = world_landmarks_i_j.x
            output[i, j, 1] = world_landmarks_i_j.y
            output[i, j, 0] = world_landmarks_i_j.z
            output[i, j, 3] = world_landmarks_i_j.visibility

    # flip y axis for compatability with openSim
    y_axis = -output[:, :, 1]
    # add minimum so ground will be 0
    output[:, :, 1] = y_axis - np.min(y_axis)
    # output[:,:,1] = y_axis
    visibility_array = output[:, :, -1]
    markers_array = output[:, :, 0:3]
    return markers_array, visibility_array


def landmarks_2_table(landmarks, landmark_names = landmark_names, time_vec=None):
    """
      Convert a list of landmark objects to two pandas dataframes containing the
      x, y, and z coordinates of each key point for each frame, and the
      visibility of each key point for each frame.

      Args:
      - landmarks: A list of landmark objects.
      - landmark_names: A list of landmark names in the same order as the
      key points in the landmark objects.
      - time_vec (array-like, optional): An array-like object containing the time
              vector associated with the landmark data. Defaults to None.

      Returns:
      - marker_df: A pandas dataframe containing the x, y, and z coordinates
      of each key point for each frame.
      - visibility_df: A pandas dataframe containing the visibility of each
      key point for each frame.
    """
    table_columns = []
    for i in range(len(landmark_names)):
        name = landmark_names[i]
        table_columns.append(name + "_x")
        table_columns.append(name + "_y")
        table_columns.append(name + "_z")

    markers_array, visibility_array = landmark_2_array(landmarks)
    # Flatten third dim (x,y,z) to the main table
    markers_array = markers_array.reshape((markers_array.shape[0], markers_array.shape[1] * 3))
    marker_df = pd.DataFrame(markers_array, columns=table_columns)
    visibility_df = pd.DataFrame(visibility_array, columns=landmark_names)
    if time_vec is not None:
        marker_df.insert(0, "t", time_vec, True)
        visibility_df.insert(0, "t", time_vec, True)

    return marker_df, visibility_df


def make_trc(df, keypoints_names, trc_path='temp.trc', frame_rate=50):
    '''
    Make Opensim compatible trc file from a dataframe with 3D coordinates

    INPUT:
    - df: pandas dataframe with 3D coordinates as columns, frame number as rows
    - keypoints_names: list of strings
    - f_range: list of two numbers. Range of frames

    OUTPUT:
    - trc file
    '''
    df = df.copy()

    # Header
    DataRate = CameraRate = OrigDataRate = frame_rate
    # TODO - add option to input diffrent DataRate , CameraRate , OrigDataRate than Frame rate
    NumFrames = len(df)
    NumMarkers = len(keypoints_names)
    df.index = df.index + 1  # np.array(range(0, f_range[1]-f_range[0])) + 1

    header_trc = ['PathFileType\t4\t(X/Y/Z)\t' + trc_path,
                  'DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames',
                  '\t'.join(map(str, [DataRate, CameraRate, NumFrames, NumMarkers, 'm', OrigDataRate, df.index[0],
                                      df.index[-1]])),  #
                  'Frame#\tTime\t' + '\t\t\t'.join(keypoints_names) + '\t\t',
                  '\t\t' + '\t'.join([f'X{i + 1}\tY{i + 1}\tZ{i + 1}' for i in range(len(keypoints_names))])]

    # Zup to Yup coordinate system
    # df = zup2yup(df)
    if 't' not in df.columns:
        # Add Frame# and Time columns
        df.insert(0, 't', df.index / frame_rate)

    # Write file
    with open(trc_path, 'w') as trc_o:
        [trc_o.write(line + '\n') for line in header_trc]
        df.to_csv(trc_o, sep='\t', index=True, header=None, line_terminator='\n')

    return trc_path
