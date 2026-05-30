from __future__ import annotations

from pathlib import Path

import cv2
import dearpygui.dearpygui as dpg
import numpy as np

from fitness_form_ai.app.config import AppConfig, SUPPORTED_EXERCISES, SUPPORTED_MODELS, TrackingConfig
from fitness_form_ai.app.session import TrackingSession

_DEFAULT_LOG_DIR = Path("logs")


class FitnessApp:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.tracking_config = TrackingConfig()
        self.session = TrackingSession(
            exercise_name=config.exercise,
            model_name=config.model,
            video_path=config.video,
            tracking_config=self.tracking_config,
        )
        self.texture_id = "video_texture"
        self.frame_data = np.zeros(
            self.tracking_config.texture_width * self.tracking_config.texture_height * 4,
            dtype=np.float32,
        )
        self._is_loading = False

    def run(self) -> None:
        dpg.create_context()
        dpg.create_viewport(title="Fitness Form Analysis AI", width=1600, height=900)

        with dpg.texture_registry(show=False):
            dpg.add_dynamic_texture(
                width=self.tracking_config.texture_width,
                height=self.tracking_config.texture_height,
                default_value=self.frame_data,
                tag=self.texture_id,
            )

        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self.on_video_select,
            tag="file_dialog_id",
            width=500,
            height=400,
        ):
            dpg.add_file_extension(".mp4", color=(0, 255, 0, 255))
            dpg.add_file_extension(".*")

        with dpg.window(tag="PrimaryWindow"):
            with dpg.group(horizontal=True):
                self._build_sidebar()
                with dpg.child_window():
                    dpg.add_image(self.texture_id)

        self.initialize_tracking()
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("PrimaryWindow", True)

        while dpg.is_dearpygui_running():
            self.process_frame()
            dpg.render_dearpygui_frame()

        self.cleanup()

    def initialize_tracking(self) -> None:
        self._is_loading = True
        self.session.initialize()
        self._set_video_status()
        self._sync_model_selector()
        self._rebuild_rule_state_panel()
        self._set_play_button_label()
        self._is_loading = False

    def process_frame(self) -> None:
        if self._is_loading:
            return

        outcome = self.session.process_next_frame()
        if outcome.image_bgr is None:
            return

        if outcome.angle is not None:
            dpg.set_value("angle_text", f"Angle: {int(outcome.angle)}°")
            cv2.putText(
                outcome.image_bgr,
                f"Angle: {int(outcome.angle)}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

        dpg.set_value("rep_count", f"Reps: {outcome.valid_reps} / {outcome.total_reps}")
        dpg.set_value("tracker_state", f"State: {outcome.tracker_state}")
        if outcome.message:
            dpg.set_value("last_msg", f"Message: {outcome.message}")

        for rule_name, value in outcome.rule_states:
            tag = f"rule_val_{rule_name}"
            if dpg.does_item_exist(tag):
                dpg.set_value(tag, f"{rule_name}: {value}")

        cv2.putText(
            outcome.image_bgr,
            f"State: {outcome.tracker_state}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            outcome.image_bgr,
            f"Reps: {outcome.valid_reps}/{outcome.total_reps}",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        texture = self.session.render_texture_data(outcome.image_bgr)
        if texture is not None:
            self.frame_data = texture
            dpg.set_value(self.texture_id, self.frame_data)

    def cleanup(self) -> None:
        self.session.cleanup()
        dpg.destroy_context()

    def on_exercise_change(self, sender: str, app_data: str, user_data: object = None) -> None:
        del sender
        del user_data
        self.session.set_exercise(app_data)
        self._set_video_status()
        self._rebuild_rule_state_panel()

    def on_model_change(self, sender: str, app_data: str, user_data: object = None) -> None:
        del sender
        del user_data
        self.session.set_model(app_data)
        self._set_video_status()
        self._sync_model_selector()
        self._rebuild_rule_state_panel()

    def on_video_select(
        self,
        sender: str,
        app_data: dict[str, str],
        user_data: object = None,
    ) -> None:
        del sender
        del user_data
        self.session.set_video_source(Path(app_data["file_path_name"]))
        self._set_video_status()
        self._set_play_button_label()

    def on_webcam_select(self, sender: object = None, app_data: object = None, user_data: object = None) -> None:
        del sender
        del app_data
        del user_data
        self.session.set_video_source(None)
        self._set_video_status()
        self._set_play_button_label()

    def on_play_pause(self, sender: object = None, app_data: object = None, user_data: object = None) -> None:
        del sender
        del app_data
        del user_data
        is_playing = self.session.toggle_playback()
        dpg.set_item_label("play_pause_btn", "Pause" if is_playing else "Play")

    def on_reset_video(self, sender: object = None, app_data: object = None, user_data: object = None) -> None:
        del sender
        del app_data
        del user_data
        self.session.reset_video()
        self._set_play_button_label()

    def on_browse_video(self, sender: object = None, app_data: object = None, user_data: object = None) -> None:
        del sender
        del app_data
        del user_data
        dpg.show_item("file_dialog_id")

    def _build_sidebar(self) -> None:
        with dpg.child_window(width=300):
            dpg.add_text("Welcome to Fitness Form AI")
            dpg.add_separator()
            dpg.add_spacer(height=10)
            dpg.add_combo(
                list(SUPPORTED_EXERCISES),
                default_value=self.config.exercise,
                label="Exercise",
                callback=self.on_exercise_change,
            )
            dpg.add_spacer(height=10)
            dpg.add_combo(
                list(SUPPORTED_MODELS),
                default_value=self.config.model,
                label="Model",
                callback=self.on_model_change,
                tag="model_selector",
            )
            dpg.add_spacer(height=10)
            dpg.add_text("Video: Webcam", tag="video_status", wrap=280)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Browse Video", callback=self.on_browse_video)
                dpg.add_button(label="Use Webcam", callback=self.on_webcam_select)
            dpg.add_spacer(height=5)
            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Pause",
                    callback=self.on_play_pause,
                    tag="play_pause_btn",
                    width=140,
                )
                dpg.add_button(label="Reset", callback=self.on_reset_video, width=140)
            dpg.add_spacer(height=20)
            dpg.add_separator()
            dpg.add_text("Tracking Stats:", color=(100, 200, 255, 255))
            dpg.add_text("Angle: 0°", tag="angle_text")
            dpg.add_spacer(height=5)
            dpg.add_text("Rule States:", color=(150, 255, 150, 255))
            with dpg.group(tag="rules_group"):
                pass
            dpg.add_spacer(height=5)
            dpg.add_text("Reps: 0 / 0", tag="rep_count")
            dpg.add_text("State: ...", tag="tracker_state")
            dpg.add_text("Message: ...", tag="last_msg", wrap=280)
            dpg.add_spacer(height=20)
            dpg.add_separator()
            dpg.add_text("Session Logger:", color=(100, 200, 255, 255))
            dpg.add_spacer(height=4)
            dpg.add_input_text(
                tag="log_dir_input",
                default_value=str(_DEFAULT_LOG_DIR),
                label="Log folder",
                width=200,
            )
            dpg.add_spacer(height=4)
            dpg.add_button(
                label="Start Logging",
                callback=self.on_log_toggle,
                tag="log_toggle_btn",
                width=-1,
            )
            dpg.add_spacer(height=4)
            dpg.add_text("Idle", tag="log_status", color=(180, 180, 180, 255), wrap=280)

    def _rebuild_rule_state_panel(self) -> None:
        if not dpg.does_item_exist("rules_group"):
            return
        dpg.delete_item("rules_group", children_only=True)
        if self.session.exercise is None:
            return
        for rule in self.session.exercise.rules:
            dpg.add_text(
                f"{rule.rule_name}: waiting for data",
                parent="rules_group",
                tag=f"rule_val_{rule.rule_name}",
                wrap=280,
            )

    def _set_video_status(self) -> None:
        label = (
            f"Video: {self.session.video_path}"
            if self.session.video_path is not None
            else "Video: Webcam"
        )
        dpg.set_value("video_status", label)

    def _set_play_button_label(self) -> None:
        if dpg.does_item_exist("play_pause_btn"):
            dpg.set_item_label(
                "play_pause_btn",
                "Pause" if self.session.is_playing else "Play",
            )

    def on_log_toggle(self, sender: object = None, app_data: object = None, user_data: object = None) -> None:
        del sender, app_data, user_data
        if self.session.is_logging:
            saved = self.session.stop_logging()
            dpg.set_item_label("log_toggle_btn", "Start Logging")
            dpg.configure_item("log_toggle_btn", enabled=True)
            if saved:
                dpg.set_value("log_status", f"Saved: {saved.name}")
                dpg.configure_item("log_status", color=(80, 220, 80, 255))
            else:
                dpg.set_value("log_status", "Nothing to save")
                dpg.configure_item("log_status", color=(180, 180, 180, 255))
        else:
            log_dir = Path(dpg.get_value("log_dir_input") or _DEFAULT_LOG_DIR)
            planned = self.session.start_logging(log_dir)
            dpg.set_item_label("log_toggle_btn", "Stop Logging")
            dpg.set_value("log_status", f"Recording → {planned.name}")
            dpg.configure_item("log_status", color=(255, 80, 80, 255))

    def _sync_model_selector(self) -> None:
        if dpg.does_item_exist("model_selector"):
            dpg.set_value("model_selector", self.session.model_name)


def run_gui(config: AppConfig) -> None:
    FitnessApp(config).run()
