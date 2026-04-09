import argparse
from gui import run_gui

def main():
    parser = argparse.ArgumentParser(description="Fitness Form Analysis AI")
    parser.add_argument("--exercise", type=str, default=None, choices=["curl", "squat", "pushup"], help="Exercise to track")
    parser.add_argument("--model", type=str, default=None, 
                        choices=["mediapipe-lite", "mediapipe-full", "mediapipe-heavy", "yolov8"], 
                        help="Pose estimation model to use")
    parser.add_argument("--video", type=str, default=None, help="Path to video file to use instead of camera. Will play in loop.")
    parser.add_argument("--gui", action="store_true", help="(Deprecated) GUI is now always launched by default")
    args = parser.parse_args()
    
    exercise = args.exercise or "curl"
    model = args.model or "mediapipe-full"
    
    print(f"Starting tracking in GUI mode...")
    run_gui(exercise, model, args.video)

if __name__ == "__main__":
    main()