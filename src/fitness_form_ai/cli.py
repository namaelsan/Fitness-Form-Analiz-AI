import argparse
from pathlib import Path

from fitness_form_ai.app.config import AppConfig, SUPPORTED_EXERCISES, SUPPORTED_MODELS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fitness Form Analysis AI")
    parser.add_argument(
        "--exercise",
        choices=SUPPORTED_EXERCISES,
        default="curl",
        help="Exercise to track.",
    )
    parser.add_argument(
        "--model",
        choices=SUPPORTED_MODELS,
        default="mediapipe-full",
        help="Pose estimation backend.",
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Optional video file path. If omitted, webcam is used.",
    )
    return parser


def main() -> int:
    from fitness_form_ai.ui.dearpygui_app import run_gui

    parser = build_parser()
    args = parser.parse_args()
    config = AppConfig(
        exercise=args.exercise,
        model=args.model,
        video=args.video,
    )
    run_gui(config)
    return 0
