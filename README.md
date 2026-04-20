# BdSL Emergency Triage Translation (Research)

This repository is an early-stage research codebase for building a **real-time, offline, bidirectional Bengali Sign Language (BdSL) translation system** focused on emergency medical triage use cases.

The current implementation is focused on **Stage 1: video data collection** with an automated OpenCV recorder.

## Current Status

Implemented:

- Automated class-wise video capture script: `scripts/collect_triage_videos.py`
- Terminal prompts for class name and number of videos
- Automatic folder creation under `triage_dataset/<CLASS_NAME>/`
- 3-second on-screen countdown before each recording
- 120-frame recording per clip at 30 FPS (~4 seconds)
- 5-second pause timer between clips
- Sequential file naming (`0.mp4`, `1.mp4`, ...)
- Emergency stop via `q`
- Black-frame protection (aborts instead of silently saving unusable clips)

Planned next stages:

1. MediaPipe keypoint extraction
2. Sequence dataset assembly (`X`, `y`, label map)
3. Model training/evaluation
4. TFLite export for mobile inference

## Repository Layout

```text
research/
├── .github/
│   └── copilot-instructions.md
├── scripts/
│   └── collect_triage_videos.py
├── triage_dataset/
│   └── <class_name>/
│       ├── 0.mp4
│       ├── 1.mp4
│       └── ...
└── README.md
```

## Requirements

- Python 3.10+
- macOS/Linux/Windows with webcam access
- OpenCV + NumPy

Install in a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip3 install opencv-python numpy
```

## How to Run Data Collection

From repository root:

```bash
python3 scripts/collect_triage_videos.py
```

You will be prompted for:

- `CLASS_NAME` (example: `emergency_doctor_need`)
- `NUMBER_OF_VIDEOS` (example: `20`)

The script will then:

1. Create `triage_dataset/CLASS_NAME` if it does not exist.
2. Continue numbering from the last existing `.mp4` index in that folder.
3. For each sample: countdown -> record -> pause -> repeat.

## Recording Timeline (Per Clip)

| Phase | Duration | What you see |
|---|---:|---|
| Get ready | 3s | Live camera feed + countdown |
| Recording | 120 frames (~4.0s @ 30 FPS) | Live feed + `RECORDING` + frame counter |
| Gap before next clip | 5s | Live feed + next-start countdown |

At any time, focus the OpenCV window and press `q` to stop safely.

## Output Format

Videos are stored here:

```text
triage_dataset/<CLASS_NAME>/<index>.mp4
```

Example:

```text
triage_dataset/emergency_doctor_need/0.mp4
triage_dataset/emergency_doctor_need/1.mp4
triage_dataset/emergency_doctor_need/2.mp4
```

If files already exist, indexing continues from the next available number.

## Camera Notes

- The script probes available camera indices/backends and chooses the first stream with valid visible frames.
- If webcam frames are black/unusable, the script raises an error instead of saving bad clips.
- On macOS, ensure Camera permission is enabled for your terminal app (Terminal/iTerm) in:
  - `System Settings -> Privacy & Security -> Camera`

## Troubleshooting

### Black preview or black output videos

1. Confirm terminal has Camera permission.
2. Close other apps that may interfere with camera routing.
3. Re-run and watch the startup log line that prints selected camera index/backend.
4. Verify lighting is sufficient (very dark scenes can look near-black).

### `ModuleNotFoundError: No module named 'cv2'`

Install dependencies in the active environment:

```bash
pip3 install opencv-python numpy
```

## Research Roadmap (High-Level)

1. Collect class-balanced raw `.mp4` samples.
2. Extract landmarks/keypoints from each clip.
3. Build fixed-length sequence tensors with label mapping.
4. Train and evaluate sequence model(s).
5. Convert final model to TFLite for mobile deployment.

## Disclaimer

This project is a research system under active development. It is **not** a medical device and should not be used for clinical decision-making in its current state.
