import math
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np


BASE_DATASET_DIR = Path.cwd() / "triage_dataset"
CAMERA_CANDIDATE_INDICES = (0, 1, 2, 3)
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 30
COUNTDOWN_SECONDS = 3.0
RECORD_TOTAL_FRAMES = 120
PAUSE_SECONDS = 5.0
WINDOW_NAME = "Triage Dataset Capture"
RESOLUTION_CANDIDATES = (
    (1280, 720),
    (960, 540),
    (640, 480),
)
WARMUP_FRAMES = 25
FRAME_BLACK_MEAN_THRESHOLD = 6.0
FRAME_BLACK_STD_THRESHOLD = 4.0


def prompt_non_empty_text(prompt: str) -> str:
    value = input(prompt).strip()
    while not value:
        print("Please enter a non-empty value.")
        value = input(prompt).strip()
    return value


def prompt_positive_int(prompt: str) -> int:
    while True:
        raw_value = input(prompt).strip()
        try:
            value = int(raw_value)
        except ValueError:
            print("Please enter a whole number.")
            continue
        if value <= 0:
            print("Please enter a number greater than 0.")
            continue
        return value


def ensure_class_directory(base_dir: Path, class_name: str) -> Path:
    class_dir = base_dir / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    return class_dir


def get_next_video_index(class_dir: Path) -> int:
    pattern = re.compile(r"^(\d+)\.mp4$")
    indices = []
    for file_path in class_dir.iterdir():
        if not file_path.is_file():
            continue
        match = pattern.match(file_path.name)
        if match:
            indices.append(int(match.group(1)))
    return max(indices, default=-1) + 1


def frame_brightness_stats(frame: np.ndarray) -> tuple[float, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return float(np.mean(gray)), float(np.std(gray))


def is_black_frame(frame: np.ndarray) -> bool:
    mean_val, std_val = frame_brightness_stats(frame)
    return mean_val <= FRAME_BLACK_MEAN_THRESHOLD and std_val <= FRAME_BLACK_STD_THRESHOLD


def open_camera() -> tuple[cv2.VideoCapture, int, str, tuple[int, int]]:
    if sys.platform == "darwin":
        backend_candidates = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    else:
        backend_candidates = [cv2.CAP_ANY]

    for camera_index in CAMERA_CANDIDATE_INDICES:
        for backend in backend_candidates:
            cap = cv2.VideoCapture(camera_index, backend)
            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
            cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

            for width, height in RESOLUTION_CANDIDATES:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

                bright_frames = 0
                for _ in range(WARMUP_FRAMES):
                    ret, frame = cap.read()
                    if not ret or frame is None or frame.size == 0:
                        time.sleep(0.02)
                        continue
                    frame = normalize_frame(frame)
                    if not is_black_frame(frame):
                        bright_frames += 1
                    time.sleep(0.01)

                if bright_frames >= 5:
                    backend_name = cap.getBackendName() if hasattr(cap, "getBackendName") else str(backend)
                    return cap, camera_index, backend_name, (width, height)

            cap.release()

    raise RuntimeError(
        "Could not find a usable webcam stream with visible frames. "
        "macOS Camera permission may still be blocked for your terminal, or your camera index/backend is unsupported."
    )


def read_camera_frame(cap: cv2.VideoCapture, retries: int = 20, require_visible: bool = False) -> np.ndarray:
    for _ in range(retries):
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            normalized = normalize_frame(frame)
            if require_visible and is_black_frame(normalized):
                time.sleep(0.01)
                continue
            return normalized
        time.sleep(0.01)
    if require_visible:
        raise RuntimeError(
            "Webcam feed is black during capture. Aborting to avoid saving unusable videos."
        )
    raise RuntimeError("Webcam frame read failed repeatedly. Recording stopped.")


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
        return cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)
    return frame


def draw_centered_text(
    frame: np.ndarray,
    text: str,
    y: int,
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    (text_width, text_height), _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness,
    )
    x = max(0, (frame.shape[1] - text_width) // 2)
    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def overlay_status(frame: np.ndarray, top_line: str, main_line: str, bottom_line: str = "") -> np.ndarray:
    canvas = frame.copy()
    draw_centered_text(canvas, top_line, 80, 0.9, (255, 255, 255), 2)
    draw_centered_text(canvas, main_line, frame.shape[0] // 2, 2.0, (0, 0, 255), 4)
    if bottom_line:
        draw_centered_text(canvas, bottom_line, frame.shape[0] - 60, 0.8, (255, 255, 0), 2)
    return canvas


def poll_for_quit() -> bool:
    key = cv2.waitKey(1) & 0xFF
    return key == ord("q")


def preview_with_timer(cap: cv2.VideoCapture, window_name: str, seconds: float, mode: str) -> bool:
    start = time.monotonic()
    while True:
        elapsed = time.monotonic() - start
        remaining = seconds - elapsed
        if remaining <= 0:
            return True

        frame = read_camera_frame(cap, require_visible=True)
        count_value = max(1, int(math.ceil(remaining)))

        if mode == "countdown":
            overlay = overlay_status(
                frame,
                "Get ready",
                str(count_value),
                "Press q to stop",
            )
        else:
            overlay = overlay_status(
                frame,
                "Next recording starts in",
                str(count_value),
                "Press q to stop",
            )

        cv2.imshow(window_name, overlay)
        if poll_for_quit():
            return False


def record_video_clip(
    cap: cv2.VideoCapture,
    output_path: Path,
    window_name: str,
    fps: int,
    total_frames: int,
) -> bool:
    frame_interval = 1.0 / float(fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (FRAME_WIDTH, FRAME_HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}.")

    start = time.monotonic()
    next_frame_time = start

    try:
        for frame_index in range(total_frames):
            now = time.monotonic()
            if now < next_frame_time:
                time.sleep(next_frame_time - now)

            frame = read_camera_frame(cap, require_visible=True)
            elapsed_seconds = (frame_index + 1) / float(fps)

            annotated = overlay_status(
                frame,
                "RECORDING",
                f"{frame_index + 1}/{total_frames} ({elapsed_seconds:.1f}s)",
                "Press q to stop",
            )
            writer.write(annotated)
            cv2.imshow(window_name, annotated)

            if poll_for_quit():
                return False

            next_frame_time += frame_interval
        return True
    finally:
        writer.release()


def main() -> int:
    class_name = prompt_non_empty_text("CLASS_NAME: ")
    number_of_videos = prompt_positive_int("NUMBER_OF_VIDEOS: ")

    class_dir = ensure_class_directory(BASE_DATASET_DIR, class_name)
    next_video_index = get_next_video_index(class_dir)

    print(f"Saving videos to: {class_dir}")
    print("Focus the OpenCV window when you want q to work.")
    print(
        f"Config -> countdown: {COUNTDOWN_SECONDS:.0f}s, recording: {RECORD_TOTAL_FRAMES} frames "
        f"({RECORD_TOTAL_FRAMES / TARGET_FPS:.1f}s at {TARGET_FPS} FPS), gap: {PAUSE_SECONDS:.0f}s"
    )

    cap, camera_index, backend_name, negotiated_resolution = open_camera()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, FRAME_WIDTH, FRAME_HEIGHT)
    print(
        f"Camera -> index: {camera_index}, backend: {backend_name}, negotiated input: "
        f"{negotiated_resolution[0]}x{negotiated_resolution[1]}"
    )

    try:
        for offset in range(number_of_videos):
            video_index = next_video_index + offset
            output_path = class_dir / f"{video_index}.mp4"

            print(f"\nRecording video {offset + 1}/{number_of_videos}: {output_path.name}")

            if not preview_with_timer(cap, WINDOW_NAME, COUNTDOWN_SECONDS, "countdown"):
                print("Capture stopped by user.")
                return 0

            if not record_video_clip(cap, output_path, WINDOW_NAME, TARGET_FPS, RECORD_TOTAL_FRAMES):
                print("Capture stopped by user.")
                return 0

            print(f"Saved: {output_path}")

            if offset < number_of_videos - 1:
                if not preview_with_timer(cap, WINDOW_NAME, PAUSE_SECONDS, "pause"):
                    print("Capture stopped by user.")
                    return 0

        print("\nDone.")
        return 0
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
