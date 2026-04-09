import time
import cv2
import traceback
import numpy as np
import dearpygui.dearpygui as dpg

from exercises import OneArmDumbellCurl, Squat, PushUp
from core.tracker import RepTracker
from util import read_landmark, calculate_angle_3d
from core.pose_model import MediaPipeModel, YOLOv8Model

EXERCISES = {
    "curl": OneArmDumbellCurl,
    "squat": Squat,
    "pushup": PushUp
}

def get_model(model_name: str):
    if model_name == "mediapipe-lite":
        return MediaPipeModel(complexity=0)
    elif model_name == "mediapipe-heavy":
        return MediaPipeModel(complexity=2)
    elif model_name == "yolov8":
        return YOLOv8Model()
    else:
        return MediaPipeModel(complexity=1)

class FitnessApp:
    def __init__(self, exercise_name="curl", model_name="mediapipe-full", video_path=None):
        self.exercise_name = exercise_name
        self.model_name = model_name
        self.video_path = video_path

        self.video_capture = None
        self.model = None
        self.exercise = None
        self.tracker = None

        self.valid_reps = 0
        self.total_reps = 0
        self.last_reason = ""
        
        self.texture_id = "video_texture"
        self.texture_width = 1200
        self.texture_height = 800
        # Initialize an empty texture
        self.frame_data = np.zeros(self.texture_width * self.texture_height * 4, dtype=np.float32)
        self._is_loading = False
        self.video_fps = 30.0
        self.video_start_time = 0
        self.is_playing = True

    def init_tracking(self):
        self._is_loading = True
        
        # Safely release video capture
        old_cap = self.video_capture
        self.video_capture = None
        if old_cap:
            try:
                old_cap.release()
            except:
                pass
        
        # Safely release model
        old_model = self.model
        self.model = None 
        if old_model:
            try:
                old_model.release()
            except:
                pass

        print(f"Initializing tracking: {self.exercise_name} with {self.model_name}")
        if self.video_path is not None and self.video_path != "":
            self.video_capture = cv2.VideoCapture(self.video_path)
            self.video_fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            if self.video_fps <= 0:
                self.video_fps = 30.0
            self.video_start_time = time.time()
            dpg.set_value("video_status", f"Video: {self.video_path}")
        else:
            self.video_capture = cv2.VideoCapture(0)
            self.video_start_time = time.time()
            dpg.set_value("video_status", "Video: Webcam")

        exercise_class = EXERCISES.get(self.exercise_name, OneArmDumbellCurl)
        self.exercise = exercise_class()
        self.tracker = RepTracker(start_phase=self.exercise.start_phase, min_rom=20.0)
        self.valid_reps = 0
        self.total_reps = 0
        self.last_reason = ""
        
        try:
            self.model = get_model(self.model_name)
        except Exception as e:
            print(f"FAILED to load {self.model_name}:")
            traceback.print_exc()
            if self.model_name != "mediapipe-full":
                print("Falling back to mediapipe-full...")
                self.model_name = "mediapipe-full"
                dpg.set_value("model_selector", "mediapipe-full")
                try:
                    self.model = get_model(self.model_name)
                    self.last_reason = "Model fallback triggered"
                except Exception as e2:
                    print(f"CRITICAL: Fallback failed: {e2}")
                    self.model = None
                    self.last_reason = "CRITICAL: All models failed"
            else:
                self.model = None
                self.last_reason = "CRITICAL: Default model failed"
        
        self._is_loading = False
        self.is_playing = True
        if dpg.does_item_exist("play_pause_btn"):
            dpg.set_item_label("play_pause_btn", "Pause")
            
    def on_play_pause(self, sender, app_data):
        if self.video_path is None or self.video_path == "":
            return # Webcam is always "playing" for now
            
        self.is_playing = not self.is_playing
        dpg.set_item_label("play_pause_btn", "Pause" if self.is_playing else "Play")
        
        if self.is_playing:
            # Adjust start time so we resume from the current frame
            current_frame = self.video_capture.get(cv2.CAP_PROP_POS_FRAMES)
            self.video_start_time = time.time() - (current_frame / self.video_fps)
            
    def on_reset_video(self, sender, app_data):
        self._is_loading = True
        
        # Always reset tracking stats
        if self.exercise:
            self.tracker = RepTracker(start_phase=self.exercise.start_phase, min_rom=20.0)
        self.valid_reps = 0
        self.total_reps = 0
        
        if self.video_path is None or self.video_path == "":
            # For webcam, nothing else to do
            self.last_reason = "Tracking Reset"
            self._is_loading = False
            return
            
        # For video file: Re-opening is much safer than seeking to 0 in FFmpeg/OpenCV
        print(f"Resetting video: {self.video_path}")
        old_cap = self.video_capture
        self.video_capture = None
        if old_cap:
            try:
                old_cap.release()
            except:
                pass
        
        try:
            self.video_capture = cv2.VideoCapture(self.video_path)
            self.video_fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            if self.video_fps <= 0:
                self.video_fps = 30.0
            self.video_start_time = time.time()
            self.is_playing = True
            if dpg.does_item_exist("play_pause_btn"):
                dpg.set_item_label("play_pause_btn", "Pause")
            self.last_reason = "Video Reset"
        except Exception as e:
            print(f"Error resetting video: {e}")
            self.last_reason = "Reset Failed"
        
        self._is_loading = False

    def on_exercise_change(self, sender, app_data):
        self.exercise_name = app_data
        self.init_tracking()

    def on_model_change(self, sender, app_data):
        self.model_name = app_data
        self.init_tracking()

    def on_video_select(self, sender, app_data):
        self.video_path = app_data['file_path_name']
        self.init_tracking()

    def on_webcam_select(self, sender, app_data):
        self.video_path = None
        self.init_tracking()

    def run(self):
        dpg.create_context()
        
        # We must initialize tracking objects after dpg context but before rendering
        # Though dpg context doesn't affect OpenCV strictly
        dpg.create_viewport(title='Fitness Form Analysis AI', width=1600, height=900)
        
        with dpg.texture_registry(show=False):
            dpg.add_dynamic_texture(width=self.texture_width, height=self.texture_height, default_value=self.frame_data, tag=self.texture_id)

        with dpg.file_dialog(directory_selector=False, show=False, callback=self.on_video_select, id="file_dialog_id", width=500, height=400):
            dpg.add_file_extension(".mp4", color=(0, 255, 0, 255))
            dpg.add_file_extension(".*")

        with dpg.window(tag="PrimaryWindow"):
            with dpg.group(horizontal=True):
                # Left panel - sidebar controls
                with dpg.child_window(width=300):
                    dpg.add_text("Welcome to Fitness Form AI")
                    dpg.add_separator()
                    
                    dpg.add_spacer(height=10)
                    dpg.add_combo(["curl", "squat", "pushup"], default_value=self.exercise_name, label="Exercise", callback=self.on_exercise_change)
                    
                    dpg.add_spacer(height=10)
                    dpg.add_combo(["mediapipe-lite", "mediapipe-full", "mediapipe-heavy", "yolov8"], default_value=self.model_name, label="Model", callback=self.on_model_change, tag="model_selector")

                    dpg.add_spacer(height=10)
                    dpg.add_text("Video: Webcam", id="video_status", wrap=280)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Browse Video", callback=lambda: dpg.show_item("file_dialog_id"))
                        dpg.add_button(label="Use Webcam", callback=self.on_webcam_select)

                    dpg.add_spacer(height=5)
                    with dpg.group(horizontal=True):
                        dpg.add_button(label="Pause", callback=self.on_play_pause, tag="play_pause_btn", width=140)
                        dpg.add_button(label="Reset", callback=self.on_reset_video, width=140)

                    dpg.add_spacer(height=20)
                    dpg.add_separator()
                    dpg.add_text("Tracking Stats:", color=(100, 200, 255, 255))
                    dpg.add_text("Angle: 0°", id="angle_text")
                    dpg.add_text("Reps: 0 / 0", id="rep_count")
                    dpg.add_text("State: ...", id="tracker_state")
                    dpg.add_text("Message: ...", id="last_msg", wrap=280)

                # Right panel - video frame
                with dpg.child_window():
                    dpg.add_image(self.texture_id)
        
        self.init_tracking() # Initialize models and videos

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("PrimaryWindow", True)

        while dpg.is_dearpygui_running():
            self.process_frame()
            dpg.render_dearpygui_frame()

        self.cleanup()

    def process_frame(self):
        if self._is_loading or not self.model:
            return

        if not self.video_capture or not self.video_capture.isOpened():
            return
        
        if self.video_path is not None and self.video_path != "":
            if not self.is_playing:
                # We still need to show the current frame even if paused
                # But don't advance the capture position
                pass
            else:
                elapsed = time.time() - self.video_start_time
                target_frame = int(elapsed * self.video_fps)
                
                # Check if we need to skip frames
                current_frame = int(self.video_capture.get(cv2.CAP_PROP_POS_FRAMES))
                if target_frame > current_frame:
                    self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        
        if not self.is_playing:
            # If paused, we just want to re-render the current frame without seeking or reading new ones
            # Wait, if we don't read, 'frame' will be old.
            # However, if we are in the loop, we might want to still show it.
            # For simplicity, if paused, we just keep showing the last 'self.frame_data' (Texture is updated below)
            # Actually, let's read the frame once when paused? 
            # No, if it's paused, just return. The texture stays.
            return
            
        ret, frame = self.video_capture.read()
        if not ret:
            if self.video_path is not None and self.video_path != "":
                # Loop video
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.video_start_time = time.time() # Reset clock for loop
                return
            else:
                return

        # Resize to texture size
        frame = cv2.resize(frame, (self.texture_width, self.texture_height))
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.model.process_image(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        landmarks = self.model.extract_landmarks(result)
        if landmarks:
            joints = [read_landmark(j, landmarks) for j in self.exercise.primary_joints]
            if None not in joints:
                angle = calculate_angle_3d(joints.copy()) 
                
                timestamp = time.time()
                rep_completed = self.tracker.add_frame(angle, timestamp, landmarks=landmarks)
                
                if rep_completed:
                    self.total_reps += 1
                    rep_data = self.tracker.extract_rep()
                    is_valid, reason = self.exercise.apply_rules(rep_data)
                    
                    if is_valid:
                        self.valid_reps += 1
                        self.last_reason = "VALID REP!"
                    else:
                        self.last_reason = f"INVALID: {reason}"
                
                dpg.set_value("angle_text", f"Angle: {int(angle)}°")

                cv2.putText(image, f"Angle: {int(angle)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            dpg.set_value("rep_count", f"Reps: {self.valid_reps} / {self.total_reps}")
            dpg.set_value("tracker_state", f"State: {self.tracker.state}")
            if self.last_reason:
                dpg.set_value("last_msg", f"Message: {self.last_reason}")

            cv2.putText(image, f"State: {self.tracker.state}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, f"Reps: {self.valid_reps}/{self.total_reps}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        self.model.draw_landmarks(image, result)

        # Convert for DPG (RGBA, float32, normalized to 0-1)
        image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        self.frame_data = (image_rgba.ravel() / 255.0).astype(np.float32)
        dpg.set_value(self.texture_id, self.frame_data)

    def cleanup(self):
        if self.video_capture:
            self.video_capture.release()
        if self.model:
            self.model.release()
        dpg.destroy_context()

def run_gui(exercise="curl", model="mediapipe-full", video=None):
    app = FitnessApp(exercise, model, video)
    app.run()
