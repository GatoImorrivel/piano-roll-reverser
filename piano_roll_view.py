"""Video preview and a canvas piano roll for the convert wizard."""

from __future__ import annotations

import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import cv2
from PIL import Image, ImageTk

from audio_player import FileAudioPlayer
from detector import Note, _white_coord, is_black_key, midi_name

NOTE_FILL = "#3d8bfd"
NOTE_FILL_BLACK = "#2a6ad4"
NOTE_ACTIVE = "#5ad0c8"
NOTE_ACTIVE_BLACK = "#3aa89f"
KEY_BG = "#0d0f12"
WHITE_KEY = "#f2f2f2"
WHITE_LIT = "#5ad0c8"
WHITE_LINE = "#c8c8c8"
BLACK_KEY = "#1a1a1a"
BLACK_LIT = "#3aa89f"
HIT_LINE = "#f0c040"
LOOKAHEAD_SEC = 2.4
KEYBOARD_H = 78


class VideoPreview(ttk.Frame):
    """Letterboxed OpenCV preview with play / seek inside a time range."""

    def __init__(self, master: tk.Misc, height: int = 320) -> None:
        super().__init__(master)
        self._cap: cv2.VideoCapture | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._after: str | None = None
        self._playing = False
        self._anchor_mono = 0.0
        self._anchor_time = 0.0
        self._on_time: callable | None = None
        self._audio = FileAudioPlayer()
        self.fps = 30.0
        self.duration = 0.0
        self.time = 0.0
        self.start = 0.0
        self.end = 0.0

        self._min_h = height
        self._label = tk.Label(self, bg="#111111")
        self._label.pack(fill=tk.BOTH, expand=True)
        self._label.bind("<Configure>", lambda _e: self._refresh())
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

    def open(self, path: Path) -> None:
        self.close()
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError("Could not open the video file.")
        self._cap = cap
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.fps = fps if fps > 1e-3 else 30.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.duration = frames / self.fps if frames else 0.0
        self.start = 0.0
        self.end = self.duration
        self._audio.open(path)
        self.seek(0.0)

    def close(self) -> None:
        self.pause()
        self._audio.close()
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def set_range(self, start: float, end: float) -> None:
        self.start = max(0.0, start)
        self.end = max(self.start, end)
        if self.time < self.start or self.time > self.end:
            self.seek(self.start)

    def set_on_time(self, callback) -> None:
        self._on_time = callback

    @property
    def has_original_audio(self) -> bool:
        return self._audio.has_original

    def set_recreated_audio(self, path: Path | None, offset: float = 0.0) -> None:
        self._audio.set_secondary(path, offset)

    def set_mix(self, original: bool, recreated: bool) -> None:
        self._audio.set_mix(original, recreated)

    @property
    def playing(self) -> bool:
        return self._playing

    def play(self) -> None:
        if self._cap is None or self._playing:
            return
        if self.time >= self.end - 1e-3:
            self.seek(self.start)
        self._playing = True
        if not self._audio.resume():
            self._audio.play(self.time, end=self.end)
        self._anchor_mono = time.monotonic()
        self._anchor_time = self.time
        self._schedule()

    def pause(self) -> None:
        self._playing = False
        self._audio.pause()
        if self._after is not None:
            self.after_cancel(self._after)
            self._after = None

    def toggle(self) -> None:
        if self._playing:
            self.pause()
        else:
            self.play()

    def seek(self, seconds: float) -> None:
        if self._cap is None:
            return
        seconds = min(max(seconds, self.start), max(self.end, self.start))
        self._cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000.0)
        ok, frame = self._cap.read()
        if ok and frame is not None:
            self.time = self._video_time()
            self._show(frame)
        else:
            self.time = seconds
        self._anchor_mono = time.monotonic()
        self._anchor_time = self.time
        if self._playing:
            self._audio.play(self.time, end=self.end)
        else:
            self._audio.stop()
        if self._on_time:
            self._on_time(self.time)

    def _video_time(self) -> float:
        if self._cap is None:
            return self.time
        msec = float(self._cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        if msec > 1.0:
            return msec / 1000.0
        pos = float(self._cap.get(cv2.CAP_PROP_POS_FRAMES) or 0.0)
        return pos / max(self.fps, 1e-3)

    def _schedule(self) -> None:
        delay = max(8, int(1000 / self.fps))
        self._after = self.after(delay, self._tick)

    def _tick(self) -> None:
        self._after = None
        if not self._playing or self._cap is None:
            return
        if self._audio.alive():
            target = min(self.end, max(self.start, self._audio.position()))
        else:
            target = self._anchor_time + (time.monotonic() - self._anchor_mono)
        if target >= self.end:
            self.pause()
            self.seek(self.end)
            return
        frame = None
        pos = self.time
        skipped = 0
        slack = 0.5 / max(self.fps, 1.0)
        if pos < target - slack:
            while skipped < 12:
                ok, nxt = self._cap.read()
                if not ok or nxt is None:
                    self.pause()
                    if frame is not None:
                        self.time = pos
                        self._show(frame)
                        if self._on_time:
                            self._on_time(self.time)
                    return
                frame = nxt
                pos = self._video_time()
                skipped += 1
                if pos >= target - slack:
                    break
        self.time = pos
        if frame is not None:
            self._show(frame)
        if self._on_time:
            self._on_time(self.time)
        self._schedule()

    def _refresh(self) -> None:
        if self._cap is None:
            return
        pos = self._cap.get(cv2.CAP_PROP_POS_FRAMES)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(pos) - 1))
        ok, frame = self._cap.read()
        if ok and frame is not None:
            self._show(frame)

    def _show(self, frame) -> None:
        w = max(self._label.winfo_width(), 16)
        h = max(self._label.winfo_height(), 16)
        fh, fw = frame.shape[:2]
        scale = min(w / fw, h / fh)
        tw, th = max(1, int(fw * scale)), max(1, int(fh * scale))
        resized = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        if tw < w or th < h:
            canvas = Image.new("RGB", (w, h), (17, 17, 17))
            canvas.paste(image, ((w - tw) // 2, (h - th) // 2))
            image = canvas
        self._photo = ImageTk.PhotoImage(image)
        self._label.configure(image=self._photo)


class PianoRollCanvas(ttk.Frame):
    """Synthesia-style falling notes over a keyboard, synced to the playhead."""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self._notes: list[Note] = []
        self._low = 21
        self._high = 108
        self._duration = 1.0
        self._playhead = 0.0

        self.canvas = tk.Canvas(self, background=KEY_BG, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Configure>", lambda _e: self.redraw())

    def set_notes(
        self, notes: list[Note], midi_low: int, midi_high: int, duration: float
    ) -> None:
        self._notes = list(notes)
        self._low = midi_low
        self._high = max(midi_high, midi_low + 1)
        self._duration = max(duration, 0.05)
        self.redraw()

    def set_playhead(self, seconds: float) -> None:
        self._playhead = max(0.0, seconds)
        self.redraw()

    def _geometry(self) -> tuple[int, int, float, float, float]:
        w = max(self.canvas.winfo_width(), 40)
        h = max(self.canvas.winfo_height(), 80)
        hit_y = h - KEYBOARD_H
        pps = max(24.0, hit_y / LOOKAHEAD_SEC)
        return w, h, hit_y, pps, 8.0

    def _key_box(self, midi: int, x0: float, x1: float) -> tuple[float, float, float]:
        whites = [m for m in range(self._low, self._high + 1) if not is_black_key(m)]
        left = _white_coord(whites[0])
        right = _white_coord(whites[-1]) + 1.0
        white_w = (x1 - x0) / max(right - left, 1e-6)
        mid = x0 + (_white_coord(midi) - left + 0.5) * white_w
        half = white_w * (0.32 if is_black_key(midi) else 0.48)
        return mid - half, mid + half, mid

    def redraw(self) -> None:
        self.canvas.delete("all")
        w, h, hit_y, pps, pad = self._geometry()
        x0, x1 = pad, w - pad
        now = self._playhead
        active = {note.pitch for note in self._notes if note.start <= now < note.end}

        self.canvas.create_rectangle(0, 0, w, hit_y, fill=KEY_BG, outline="")
        for note in self._notes:
            if note.end < now or note.start > now + LOOKAHEAD_SEC + 0.25:
                continue
            xa, xb, _mid = self._key_box(note.pitch, x0, x1)
            y_bottom = hit_y - (note.start - now) * pps
            y_top = hit_y - (note.end - now) * pps
            y_bottom = min(hit_y, y_bottom)
            y_top = max(0.0, y_top)
            if y_bottom - y_top < 2:
                continue
            playing = note.start <= now < note.end
            black = is_black_key(note.pitch)
            if playing:
                fill = NOTE_ACTIVE_BLACK if black else NOTE_ACTIVE
            else:
                fill = NOTE_FILL_BLACK if black else NOTE_FILL
            self.canvas.create_rectangle(
                xa + 1,
                y_top,
                xb - 1,
                y_bottom,
                fill=fill,
                outline="",
            )

        self.canvas.create_line(0, hit_y, w, hit_y, fill=HIT_LINE, width=2)
        self._draw_keyboard(x0, x1, hit_y, h, active)

    def _draw_keyboard(
        self, x0: float, x1: float, y0: float, y1: float, active: set[int]
    ) -> None:
        midis = list(range(self._low, self._high + 1))
        whites = [m for m in midis if not is_black_key(m)]
        blacks = [m for m in midis if is_black_key(m)]
        black_bottom = y0 + (y1 - y0) * 0.58
        for midi in whites:
            xa, xb, mid = self._key_box(midi, x0, x1)
            lit = midi in active
            self.canvas.create_rectangle(
                xa, y0, xb, y1, fill=WHITE_LIT if lit else WHITE_KEY, outline=WHITE_LINE
            )
            if midi % 12 == 0:
                self.canvas.create_text(
                    mid,
                    y1 - 10,
                    text=midi_name(midi),
                    fill="#333333",
                    font=("TkDefaultFont", 8),
                )
        for midi in blacks:
            xa, xb, _mid = self._key_box(midi, x0, x1)
            lit = midi in active
            self.canvas.create_rectangle(
                xa, y0, xb, black_bottom, fill=BLACK_LIT if lit else BLACK_KEY, outline="#000000"
            )
