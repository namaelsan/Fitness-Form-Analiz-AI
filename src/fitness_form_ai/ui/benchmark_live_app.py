"""Live benchmark viewer — runs the benchmark in a background thread and
displays each frame with pose landmarks and running stats in a DearPyGui window."""

from __future__ import annotations

import queue
import threading
from pathlib import Path

import cv2
import numpy as np
import dearpygui.dearpygui as dpg

from fitness_form_ai.evaluation.benchmark import LiveFrameUpdate, run_all_benchmarks
from fitness_form_ai.evaluation.compare import compare_results
from fitness_form_ai.evaluation.report import generate_full_report


_TEXTURE_W = 960
_TEXTURE_H = 540


class BenchmarkLiveApp:
    def __init__(
        self,
        model_names: list[str],
        video_paths: list[Path],
        exercise_name: str,
        output_dir: Path,
        reference_model: str,
        resize: tuple[int, int] = (640, 480),
    ) -> None:
        self._model_names = model_names
        self._video_paths = video_paths
        self._exercise_name = exercise_name
        self._output_dir = output_dir
        self._reference_model = reference_model
        self._resize = resize

        # Queue holds at most 2 frames; older ones are dropped so the GUI
        # never lags behind the benchmark thread.
        self._frame_queue: queue.Queue[LiveFrameUpdate] = queue.Queue(maxsize=2)
        self._done = threading.Event()
        self._results_ready: list = []
        self._error: Exception | None = None

        self._texture_tag = "bench_live_texture"
        self._blank = np.zeros(_TEXTURE_W * _TEXTURE_H * 4, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        dpg.create_context()
        dpg.create_viewport(
            title="Fitness Form AI — Live Benchmark",
            width=_TEXTURE_W + 360,
            height=_TEXTURE_H + 60,
        )

        with dpg.texture_registry(show=False):
            dpg.add_dynamic_texture(
                width=_TEXTURE_W,
                height=_TEXTURE_H,
                default_value=self._blank,
                tag=self._texture_tag,
            )

        with dpg.window(tag="BenchLive"):
            with dpg.group(horizontal=True):
                # ---- video panel ----
                with dpg.child_window(width=_TEXTURE_W, height=_TEXTURE_H + 10, border=False):
                    dpg.add_image(self._texture_tag)

                # ---- stats sidebar ----
                with dpg.child_window(width=340, border=True):
                    dpg.add_text("Live Benchmark", color=(100, 200, 255, 255))
                    dpg.add_separator()
                    dpg.add_spacer(height=6)

                    dpg.add_text("Overall progress", color=(180, 180, 180, 255))
                    dpg.add_text("Job: —", tag="live_job")
                    with dpg.group(horizontal=True):
                        dpg.add_text("", tag="live_progress_bar_lbl")
                    dpg.add_progress_bar(
                        default_value=0.0,
                        tag="live_progress_bar",
                        width=-1,
                        overlay="0 %",
                    )
                    dpg.add_spacer(height=10)

                    dpg.add_text("Current job", color=(180, 180, 180, 255))
                    dpg.add_text("Model:    —", tag="live_model")
                    dpg.add_text("Video:    —", tag="live_video")
                    dpg.add_text("Exercise: —", tag="live_exercise")
                    dpg.add_spacer(height=10)

                    dpg.add_text("Frame stats", color=(180, 180, 180, 255))
                    dpg.add_text("Frame:  —", tag="live_frame")
                    dpg.add_text("FPS:    —", tag="live_fps")
                    dpg.add_text("Detection: —", tag="live_detection")
                    dpg.add_spacer(height=10)

                    dpg.add_text("Repetitions", color=(180, 180, 180, 255))
                    dpg.add_text("Reps:  —", tag="live_reps")
                    dpg.add_spacer(height=20)

                    dpg.add_separator()
                    dpg.add_spacer(height=6)
                    dpg.add_text("Waiting for benchmark…", tag="live_status",
                                 color=(255, 220, 80, 255), wrap=320)

        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_primary_window("BenchLive", True)

        # Start benchmark worker
        worker = threading.Thread(target=self._benchmark_worker, daemon=True)
        worker.start()

        # GUI render loop
        while dpg.is_dearpygui_running():
            self._consume_queue()
            if self._done.is_set() and self._frame_queue.empty():
                self._on_done()
                break
            dpg.render_dearpygui_frame()

        # Show final state briefly, then auto-close
        import time
        deadline = time.monotonic() + 2.0
        while dpg.is_dearpygui_running() and time.monotonic() < deadline:
            dpg.render_dearpygui_frame()

        dpg.destroy_context()

        # Write reports after the window closes
        if self._results_ready and self._error is None:
            print("\nGenerating reports…")
            comparison = compare_results(
                self._results_ready, reference_model=self._reference_model
            )
            generate_full_report(self._results_ready, comparison, self._output_dir)

        if self._error is not None:
            raise self._error

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _benchmark_worker(self) -> None:
        try:
            results = run_all_benchmarks(
                model_names=self._model_names,
                video_paths=self._video_paths,
                exercise_name=self._exercise_name,
                resize=self._resize,
                verbose=True,
                live_callback=self._push_frame,
            )
            self._results_ready = results
        except Exception as exc:
            self._error = exc
        finally:
            self._done.set()

    def _push_frame(self, update: LiveFrameUpdate) -> None:
        """Called from the benchmark thread — drop frame if GUI can't keep up."""
        try:
            self._frame_queue.put_nowait(update)
        except queue.Full:
            pass

    def _consume_queue(self) -> None:
        try:
            update = self._frame_queue.get_nowait()
        except queue.Empty:
            return

        # Update texture
        frame = cv2.resize(update.frame_bgr, (_TEXTURE_W, _TEXTURE_H))
        rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        texture_data = (rgba.ravel() / 255.0).astype(np.float32)
        dpg.set_value(self._texture_tag, texture_data)

        # Overall progress
        job_pct = update.job_idx / update.total_jobs if update.total_jobs else 0.0
        # Within-job frame progress
        if update.total_frames > 0:
            frame_pct = update.frame_idx / update.total_frames
        else:
            frame_pct = 0.0
        # Combined: finished jobs + fraction of current job
        combined = (update.job_idx - 1 + frame_pct) / update.total_jobs if update.total_jobs else 0.0

        dpg.set_value("live_job", f"Job: {update.job_idx} / {update.total_jobs}")
        dpg.set_value("live_progress_bar", combined)
        dpg.configure_item("live_progress_bar", overlay=f"{combined * 100:.0f} %")

        # Current job
        dpg.set_value("live_model", f"Model:    {update.model_name}")
        dpg.set_value("live_video", f"Video:    {update.video_name}")
        dpg.set_value("live_exercise", f"Exercise: {update.exercise_name}")

        # Frame stats
        frame_label = (
            f"Frame:  {update.frame_idx + 1} / {update.total_frames}"
            if update.total_frames > 0
            else f"Frame:  {update.frame_idx + 1}"
        )
        dpg.set_value("live_frame", frame_label)
        dpg.set_value("live_fps", f"FPS:    {update.mean_fps:.1f}")
        dpg.set_value("live_detection", f"Detection: {update.detection_rate:.1%}")

        # Reps
        dpg.set_value("live_reps",
                      f"Reps:  {update.valid_reps} valid / {update.total_reps} total")

        # Status
        dpg.set_value("live_status", "Benchmarking…")

    def _on_done(self) -> None:
        if self._error:
            dpg.set_value("live_status", f"Error: {self._error}")
            dpg.configure_item("live_status", color=(255, 80, 80, 255))
        else:
            dpg.set_value("live_status",
                          "Done! Writing reports…\nWindow closing automatically.")
            dpg.configure_item("live_status", color=(80, 255, 80, 255))
