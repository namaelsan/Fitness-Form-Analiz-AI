# Fitness Form Analysis AI

Realtime exercise form analysis with pose-estimation backends, repetition tracking, and a Dear PyGui desktop interface.

---

## Installation

```bash
# Core app
pip install -e .

# With evaluation/benchmarking support (matplotlib, seaborn)
pip install -e ".[eval]"

# With development tools (pytest, ruff)
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

---

## Commands

| Command | Purpose |
|---|---|
| `fitness-form-ai` | Live GUI — webcam or video file |
| `fitness-form-evaluate` | Offline batch benchmark |

---

## `fitness-form-ai` — Live GUI

Opens the DearPyGui interface for real-time form feedback.

```
fitness-form-ai [--exercise EXERCISE] [--model MODEL] [--video PATH]
```

| Flag | Choices / Default | Description |
|---|---|---|
| `--exercise` | `curl`, `squat`, `deadlift`, `shoulder_press`, `lateral_raise` · **default: `curl`** | Exercise to track. |
| `--model` | `mediapipe-lite`, `mediapipe-full`, `mediapipe-heavy`, `yolov8`, `movenet-lightning`, `movenet-thunder` · **default: `mediapipe-full`** | Pose estimation backend. |
| `--video` | path to video file · **default: webcam** | Video file to analyse. Omit to use the webcam. |

### Examples

```bash
# Webcam, biceps curl, default model
fitness-form-ai

# Webcam, squat with heavyweight model
fitness-form-ai --exercise squat --model mediapipe-heavy

# Analyse a saved video file
fitness-form-ai --exercise curl --model yolov8 --video my_workout.mp4
```

---

## `fitness-form-evaluate` — Offline Benchmark

Batch-processes videos with one or more pose models, measures per-frame latency, detection rate, rep counts, and rule pass/fail — then writes a full report.

Requires the `[eval]` extra (`pip install -e ".[eval]"`).

```
fitness-form-evaluate (--videos PATH [PATH …] | --video-set SPEC)
                      [--exercise EXERCISE]
                      [--models MODEL [MODEL …]]
                      [--reference MODEL]
                      [--output-dir DIR]
                      [--resize W H]
                      [--examples-dir DIR]
```

`--videos` and `--video-set` are mutually exclusive; one is required.

| Flag | Choices / Default | Description |
|---|---|---|
| `--videos` | one or more file paths | Explicit video files to evaluate. |
| `--video-set` | `EXERCISE/TYPE`, `EXERCISE`, or `all` | Auto-discover videos from the examples folder (see below). |
| `--examples-dir` | path · **default: `examples/`** | Root of the examples folder used by `--video-set`. |
| `--exercise` | same choices as above · **default: `squat`** | Exercise whose rules are applied during benchmarking. |
| `--models` | one or more model names · **default: all models** | Models to benchmark. |
| `--reference` | model name · **default: `mediapipe-heavy`** | Reference model for the confusion matrix. Auto-added to `--models` if not listed. |
| `--output-dir` | path · **default: `./results`** | Directory where reports and CSVs are written. Created if it does not exist. |
| `--resize` | two integers `W H` · **default: `640 480`** | Resize every frame to this resolution before inference. |
| `--live` | flag · **default: off** | Open a GUI window and watch the benchmark run in real time. Reports are still written after the window is closed. |

### `--video-set` spec

The examples folder is expected to follow this layout:

```
examples/
  biceps/
    casual/
    proper/
  squat/
    proper/
  …
```

| Spec | Videos included |
|---|---|
| `biceps/casual` | Only `examples/biceps/casual/*` |
| `biceps/proper` | Only `examples/biceps/proper/*` |
| `biceps` | All videos under `examples/biceps/` |
| `squat` | All videos under `examples/squat/` |
| `all` | Every video under `examples/` |

### Examples

```bash
# Watch the benchmark live in a GUI window
fitness-form-evaluate --video-set biceps/casual --exercise curl --live

# Only biceps casual videos, default model set
fitness-form-evaluate --video-set biceps/casual --exercise curl

# Biceps casual + proper, two specific models
fitness-form-evaluate --video-set biceps --exercise curl \
  --models mediapipe-full yolov8

# Every example video across all exercises
fitness-form-evaluate --video-set all --exercise curl --output-dir results/full

# Explicit file paths (bypasses examples folder)
fitness-form-evaluate --videos rec1.mp4 rec2.mp4 --exercise squat

# Custom resize and output directory
fitness-form-evaluate --video-set squat/proper --exercise squat \
  --resize 1280 720 --output-dir results/hd

# Different reference model for the confusion matrix
fitness-form-evaluate --video-set biceps/casual --exercise curl \
  --reference mediapipe-full --models mediapipe-lite mediapipe-full
```

---

## Supported exercises and models

**Exercises:** `curl`, `squat`, `deadlift`, `shoulder_press`, `lateral_raise`

**Models:**

| Name | Notes |
|---|---|
| `mediapipe-lite` | Fastest, lower accuracy |
| `mediapipe-full` | Balanced (GUI default) |
| `mediapipe-heavy` | Most accurate MediaPipe variant; benchmark reference default |
| `yolov8` | YOLOv8-pose |
| `movenet-lightning` | MoveNet Lightning — very fast |
| `movenet-thunder` | MoveNet Thunder — higher accuracy |
