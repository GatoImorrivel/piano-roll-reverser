"""OpenCV overlay so you can see keyboard mapping and live key hits."""

from __future__ import annotations

import argparse
import tempfile
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

import cv2

from detector import (
    DetectionError,
    Key,
    _build_keyboard,
    _calibrate_baselines,
    _calibration_frame,
    _detect_black_centers,
    _sample_rect,
    frame_key_states,
    midi_name,
)
from downloader import DownloadError, download_youtube, is_url, validate_local_video

WINDOW = "piano-roll debug"
HELP = "space pause | , . step | [ ] speed | q quit"


def ask_debug_source() -> str | None:
    """Small window: YouTube URL or local file."""
    root = tk.Tk()
    root.title("Debug video")
    root.resizable(True, False)
    root.minsize(460, 140)

    chosen: dict[str, str] = {}
    url_var = tk.StringVar()
    file_var = tk.StringVar(value="No file selected")
    local_path: list[str] = [""]

    def pick_file() -> None:
        path = filedialog.askopenfilename(
            parent=root,
            title="Choose a piano tutorial video",
            filetypes=[
                ("Video files", "*.mp4 *.mkv *.webm *.mov *.avi *.m4v"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        local_path[0] = path
        url_var.set("")
        file_var.set(Path(path).name)

    def use_url() -> None:
        url = url_var.get().strip()
        if url:
            chosen["source"] = url
            root.destroy()
            return
        if local_path[0]:
            chosen["source"] = local_path[0]
            root.destroy()

    frame = ttk.Frame(root, padding=16)
    frame.pack(fill=tk.BOTH, expand=True)
    ttk.Label(frame, text="YouTube URL").pack(anchor=tk.W)
    ttk.Entry(frame, textvariable=url_var).pack(fill=tk.X, pady=(2, 8))
    ttk.Label(frame, text="or").pack(anchor=tk.W)
    row = ttk.Frame(frame)
    row.pack(fill=tk.X, pady=(4, 12))
    ttk.Button(row, text="Choose local video…", command=pick_file).pack(side=tk.LEFT)
    ttk.Label(row, textvariable=file_var).pack(side=tk.LEFT, padx=(10, 0))
    btns = ttk.Frame(frame)
    btns.pack(fill=tk.X)
    ttk.Button(btns, text="Open debug", command=use_url).pack(side=tk.LEFT)
    ttk.Button(btns, text="Cancel", command=root.destroy).pack(side=tk.LEFT, padx=(8, 0))
    root.bind("<Return>", lambda _e: use_url())
    root.mainloop()
    return chosen.get("source")


def run_debug(source: str) -> None:
    print("Loading video…")
    video_path = _resolve_source(source)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise DetectionError("Could not open the video file.")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cal = _calibration_frame(cap, frame_count)
        keys = _build_keyboard(cal)
        _calibrate_baselines(cap, keys, frame_count)
        gray = cv2.cvtColor(cal, cv2.COLOR_BGR2GRAY)
        y0 = min(k.y0 for k in keys)
        y1 = max(k.y1 for k in keys)
        x0 = min(k.x0 for k in keys)
        x1 = max(k.x1 for k in keys)
        black_centers = _detect_black_centers(gray, x0, x1, y0, y1)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        paused = False
        delay = max(1, int(1000 / fps))
        index = 0
        last: np.ndarray | None = None

        print(f"Keys: {len(keys)}  {midi_name(keys[0].midi)}–{midi_name(keys[-1].midi)}")
        print(HELP)

        while True:
            if not paused or last is None:
                ok, frame = cap.read()
                if not ok or frame is None:
                    if last is None:
                        raise DetectionError("Could not read frames from the video.")
                    paused = True
                    frame = last
                else:
                    last = frame
                    index += 1
            else:
                frame = last

            ons, _strength = frame_key_states(frame, keys)
            canvas = _draw_overlay(frame, keys, ons, black_centers, index, fps)
            shown = _fit_display(canvas)
            cv2.imshow(WINDOW, shown)

            key = cv2.waitKey(0 if paused else delay) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
            elif key == ord(",") and index > 1:
                index = max(0, index - 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                paused = True
                last = None
            elif key == ord("."):
                paused = False
            elif key == ord("["):
                delay = min(500, delay * 2)
            elif key == ord("]"):
                delay = max(1, delay // 2)
    finally:
        cap.release()
        cv2.destroyAllWindows()


def _resolve_source(source: str) -> Path:
    if is_url(source):
        dest = Path(tempfile.mkdtemp(prefix="piano-roll-debug-"))
        print("Downloading YouTube video…")
        path, title = download_youtube(
            source, dest, lambda frac: print(f"\rDownload {frac * 100:5.1f}%", end="", flush=True)
        )
        print(f"\nDownloaded {title}")
        return path
    return validate_local_video(source)


def _draw_overlay(
    frame: np.ndarray,
    keys: list[Key],
    ons: list[bool],
    black_centers: list[float],
    index: int,
    fps: float,
) -> np.ndarray:
    canvas = frame.copy()
    h, w = canvas.shape[:2]
    y0 = min(k.y0 for k in keys)
    y1 = max(k.y1 for k in keys)
    x0 = min(max(0, k.x0) for k in keys)
    x1 = max(k.x1 for k in keys)

    cv2.rectangle(canvas, (x0, y0), (min(w - 1, x1), min(h - 1, y1)), (80, 220, 80), 2)

    for cx in black_centers:
        cv2.line(canvas, (int(cx), y0), (int(cx), y0 + 12), (255, 180, 80), 1)

    active_names: list[str] = []
    for key, on in zip(keys, ons):
        color = (0, 220, 255) if on else ((180, 80, 180) if key.is_black else (200, 180, 80))
        pt1 = (max(0, key.x0), max(0, key.y0))
        pt2 = (min(w - 1, key.x1), min(h - 1, key.y1))
        cv2.rectangle(canvas, pt1, pt2, color, 1 if on else 1)
        sx0, sy0, sx1, sy1 = _sample_rect(key, w, h)
        cv2.rectangle(canvas, (sx0, sy0), (sx1, sy1), (0, 255, 255) if on else (40, 40, 240), -1)
        if on or (not key.is_black and key.midi % 12 == 0):
            label = midi_name(key.midi)
            cv2.putText(
                canvas,
                label,
                (max(0, key.x0), max(16, key.y0 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.32,
                color,
                1,
                cv2.LINE_AA,
            )
        if not key.is_black:
            letter = midi_name(key.midi)
            letter = letter[:-1] if letter[-1].isdigit() else letter
            cv2.putText(
                canvas,
                letter,
                (max(0, key.mid_x - 4), min(h - 4, key.y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (180, 60, 20) if not on else (0, 90, 180),
                1,
                cv2.LINE_AA,
            )
        if on:
            active_names.append(midi_name(key.midi))

    hud = (
        f"t={index / fps:7.2f}s  frame={index}  "
        f"keys={len(keys)} {midi_name(keys[0].midi)}-{midi_name(keys[-1].midi)}  "
        f"on={','.join(active_names) or '-'}  |  {HELP}"
    )
    cv2.rectangle(canvas, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.putText(canvas, hud, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
    return canvas


def _fit_display(frame: np.ndarray, max_w: int = 1280) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_w:
        return frame
    scale = max_w / w
    return cv2.resize(frame, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)


def add_debug_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--debug",
        nargs="?",
        const="",
        metavar="VIDEO",
        help="Open the OpenCV debug viewer. Pass a local video or YouTube URL.",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Piano-roll debug viewer")
    add_debug_args(parser)
    parser.add_argument("video", nargs="?", help="Local video or YouTube URL")
    args = parser.parse_args(argv)
    source = args.debug or args.video
    if not source:
        source = ask_debug_source() or ""
    if not source:
        raise SystemExit("Usage: python app.py --debug VIDEO_OR_URL")
    try:
        run_debug(source)
    except (DetectionError, DownloadError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
