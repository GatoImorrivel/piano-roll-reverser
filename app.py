#!/usr/bin/env python3
"""Tkinter wizard: pick video → trim start/end → compare video + piano roll."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from audio_player import ensure_pcm_wav
from detector import DetectionError, DetectionResult, detect_notes
from downloader import (
    AuthRequired,
    DownloadError,
    ask_youtube_login,
    download_youtube,
    has_saved_cookies,
    is_url,
    validate_local_video,
)
from midi_synth import render_notes_wav
from midi_writer import write_midi
from piano_roll_view import PianoRollCanvas, VideoPreview


def _fmt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    mins, secs = divmod(int(seconds), 60)
    return f"{mins:d}:{secs:02d}.{int((seconds - int(seconds)) * 10):d}"


def _safe_filename(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in " -_." else "_" for ch in name)
    cleaned = cleaned.strip(" ._") or "output"
    return cleaned[:80]


class PianoRollApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Piano Roll → MIDI")
        self.minsize(960, 640)
        self.geometry("1280x800")

        self._tempdir: Path | None = None
        self._video_path: Path | None = None
        self._midi_path: Path | None = None
        self._recreated_path: Path | None = None
        self._suggested_name = "output.mid"
        self._title = "output"
        self._result: DetectionResult | None = None
        self._detect_range: tuple[float, float] | None = None
        self._busy = False
        self._last_progress_ui = 0.0
        self._local_path: Path | None = None
        self._syncing = False

        self._build()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        header = ttk.Frame(self, padding=(16, 12, 16, 0))
        header.grid(row=0, column=0, sticky="ew")
        ttk.Label(header, text="Piano Roll → MIDI", font=("TkDefaultFont", 14, "bold")).pack(
            anchor=tk.W
        )
        self.step_var = tk.StringVar(value="1 / 3  Choose a video")
        ttk.Label(header, textvariable=self.step_var).pack(anchor=tk.W)

        self._body = ttk.Frame(self, padding=16)
        self._body.grid(row=1, column=0, sticky="nsew")
        self._body.columnconfigure(0, weight=1)
        self._body.rowconfigure(0, weight=1)

        self._pick = self._build_pick()
        self._trim = self._build_trim()
        self._compare = self._build_compare()
        for frame in (self._pick, self._trim, self._compare):
            frame.grid(row=0, column=0, sticky="nsew")

        footer = ttk.Frame(self, padding=(16, 8, 16, 16))
        footer.grid(row=2, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(footer, mode="determinate", maximum=100)
        self.progress.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.status_var = tk.StringVar(value="Waiting for a video")
        ttk.Label(footer, textvariable=self.status_var).grid(row=1, column=0, sticky="w")

        self._show("pick")

    def _build_pick(self) -> ttk.Frame:
        frame = ttk.Frame(self._body)
        ttk.Label(
            frame,
            text="Turn a Synthesia-style piano tutorial into a MIDI file.",
        ).pack(anchor=tk.W, pady=(0, 12))

        ttk.Label(frame, text="YouTube URL").pack(anchor=tk.W)
        self.url_var = tk.StringVar()
        ttk.Entry(frame, textvariable=self.url_var).pack(fill=tk.X, pady=(2, 8))
        self.url_var.trace_add("write", self._on_url_typed)

        ttk.Label(frame, text="or").pack(anchor=tk.W)
        file_row = ttk.Frame(frame)
        file_row.pack(fill=tk.X, pady=(4, 16))
        ttk.Button(file_row, text="Choose local video…", command=self._choose_video).pack(
            side=tk.LEFT
        )
        self.file_var = tk.StringVar(value="No file selected")
        ttk.Label(file_row, textvariable=self.file_var).pack(side=tk.LEFT, padx=(10, 0))

        next_row = ttk.Frame(frame)
        next_row.pack(anchor=tk.W)
        ttk.Button(next_row, text="Next", command=self._pick_next).pack(side=tk.LEFT)
        ttk.Button(
            next_row, text="Sign in to YouTube…", command=self._sign_in_youtube
        ).pack(side=tk.LEFT, padx=(8, 0))
        self.youtube_status_var = tk.StringVar()
        ttk.Label(next_row, textvariable=self.youtube_status_var).pack(
            side=tk.LEFT, padx=(10, 0)
        )
        self._refresh_youtube_status()
        return frame

    def _build_trim(self) -> ttk.Frame:
        frame = ttk.Frame(self._body)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        self.trim_video = VideoPreview(frame, height=360)
        self.trim_video.grid(row=0, column=0, sticky="nsew")
        self.trim_video.set_on_time(self._on_trim_time)

        controls = ttk.Frame(frame)
        controls.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        controls.columnconfigure(1, weight=1)

        self.trim_play_btn = ttk.Button(controls, text="Play", command=self._toggle_trim)
        self.trim_play_btn.grid(row=0, column=0, padx=(0, 8))
        self.trim_seek = ttk.Scale(
            controls, from_=0, to=1, orient=tk.HORIZONTAL, command=self._on_trim_seek
        )
        self.trim_seek.grid(row=0, column=1, sticky="ew")
        self.trim_seek.bind("<ButtonPress-1>", lambda _e: self.trim_video.pause())
        self.trim_time_var = tk.StringVar(value="0:00.0")
        ttk.Label(controls, textvariable=self.trim_time_var, width=10).grid(
            row=0, column=2, padx=(8, 0)
        )

        marks = ttk.Frame(frame)
        marks.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        marks.columnconfigure(1, weight=1)
        marks.columnconfigure(3, weight=1)
        ttk.Button(marks, text="Set start", command=self._set_start_here).grid(row=0, column=0)
        self.start_var = tk.DoubleVar(value=0.0)
        self.start_scale = ttk.Scale(
            marks, from_=0, to=1, variable=self.start_var, command=self._on_start_moved
        )
        self.start_scale.grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(marks, text="Set end", command=self._set_end_here).grid(row=0, column=2)
        self.end_var = tk.DoubleVar(value=1.0)
        self.end_scale = ttk.Scale(
            marks, from_=0, to=1, variable=self.end_var, command=self._on_end_moved
        )
        self.end_scale.grid(row=0, column=3, sticky="ew", padx=8)
        self.range_var = tk.StringVar(value="Range 0:00.0 – 0:00.0")
        ttk.Label(marks, textvariable=self.range_var).grid(
            row=1, column=0, columnspan=4, sticky="w", pady=(6, 0)
        )

        nav = ttk.Frame(frame)
        nav.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        ttk.Button(nav, text="Back", command=lambda: self._show("pick")).pack(side=tk.LEFT)
        ttk.Button(nav, text="Next — extract MIDI", command=self._trim_next).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        return frame

    def _build_compare(self) -> ttk.Frame:
        frame = ttk.Frame(self._body)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=2)
        frame.rowconfigure(1, weight=3)

        self.compare_video = VideoPreview(frame, height=280)
        self.compare_video.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        self.compare_video.set_on_time(self._on_compare_time)

        self.roll = PianoRollCanvas(frame)
        self.roll.grid(row=1, column=0, sticky="nsew")

        bar = ttk.Frame(frame)
        bar.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        bar.columnconfigure(1, weight=1)
        self.compare_play_btn = ttk.Button(bar, text="Play", command=self._toggle_compare)
        self.compare_play_btn.pack(side=tk.LEFT)
        self.compare_seek = ttk.Scale(
            bar, from_=0, to=1, orient=tk.HORIZONTAL, command=self._on_compare_seek
        )
        self.compare_seek.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        self.compare_seek.bind("<ButtonPress-1>", lambda _e: self.compare_video.pause())
        self.compare_time_var = tk.StringVar(value="0:00.0")
        ttk.Label(bar, textvariable=self.compare_time_var, width=10).pack(side=tk.LEFT)

        nav = ttk.Frame(frame)
        nav.grid(row=3, column=0, sticky="ew", pady=(10, 0))
        ttk.Button(nav, text="Back", command=self._compare_back).pack(side=tk.LEFT)
        self.orig_audio_var = tk.BooleanVar(value=True)
        self.rec_audio_var = tk.BooleanVar(value=True)
        self.orig_audio_btn = ttk.Checkbutton(
            nav,
            text="Original Audio",
            variable=self.orig_audio_var,
            command=self._on_mix_changed,
        )
        self.orig_audio_btn.pack(side=tk.LEFT, padx=(16, 0))
        self.rec_audio_btn = ttk.Checkbutton(
            nav,
            text="Recreated Audio",
            variable=self.rec_audio_var,
            command=self._on_mix_changed,
        )
        self.rec_audio_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.save_btn = ttk.Button(nav, text="Save MIDI…", command=self._save_midi)
        self.save_btn.pack(side=tk.RIGHT)
        return frame

    def _show(self, name: str) -> None:
        self.trim_video.pause()
        self.compare_video.pause()
        pages = {"pick": self._pick, "trim": self._trim, "compare": self._compare}
        labels = {
            "pick": "1 / 3  Choose a video",
            "trim": "2 / 3  Choose start and end",
            "compare": "3 / 3  Compare video and MIDI",
        }
        pages[name].tkraise()
        self.step_var.set(labels[name])

    def _on_url_typed(self, *_args: object) -> None:
        if self.url_var.get().strip():
            self._local_path = None
            self.file_var.set("No file selected")

    def _choose_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose a piano tutorial video",
            filetypes=[
                ("Video files", "*.mp4 *.mkv *.webm *.mov *.avi *.m4v"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            self._local_path = validate_local_video(path)
        except DownloadError as exc:
            messagebox.showerror("Invalid video", str(exc), parent=self)
            return
        self.url_var.set("")
        self.file_var.set(self._local_path.name)

    def _refresh_youtube_status(self) -> None:
        if has_saved_cookies():
            self.youtube_status_var.set("YouTube: signed in")
        else:
            self.youtube_status_var.set("YouTube: not signed in")

    def _sign_in_youtube(self) -> None:
        ask_youtube_login(self)
        self._refresh_youtube_status()

    def _pick_next(self) -> None:
        if self._busy:
            return
        url = self.url_var.get().strip()
        if url:
            source, use_url = url, True
        elif self._local_path is not None:
            source, use_url = str(self._local_path), False
        else:
            messagebox.showinfo(
                "No video",
                "Paste a YouTube URL or choose a local video first.",
                parent=self,
            )
            return
        self._busy = True
        self._reset_tempdir()
        self.progress["value"] = 0
        self.status_var.set("Starting…")
        threading.Thread(
            target=self._download_worker,
            args=(source, use_url, False),
            daemon=True,
        ).start()

    def _download_worker(self, source: str, use_url: bool, retried_auth: bool) -> None:
        try:
            assert self._tempdir is not None
            if use_url:
                if not is_url(source):
                    raise DownloadError("Enter a valid http(s) YouTube URL.")
                video_path, title = download_youtube(
                    source, self._tempdir, lambda frac: self._progress_from_thread("download", frac)
                )
            else:
                video_path = validate_local_video(source)
                title = video_path.stem
            self.after(0, lambda: self.status_var.set("Preparing audio…"))
            ensure_pcm_wav(video_path)
        except AuthRequired as exc:
            if retried_auth:
                self.after(0, self._on_error, str(exc))
                return
            self.after(0, self._ask_login_and_retry, source, str(exc))
            return
        except (DownloadError, OSError) as exc:
            self.after(0, self._on_error, str(exc))
            return
        except Exception as exc:
            self.after(0, self._on_error, f"Unexpected error: {exc}")
            return
        self.after(0, self._on_video_ready, video_path, title)

    def _ask_login_and_retry(self, source: str, message: str) -> None:
        self._busy = False
        self.progress["value"] = 0
        self.status_var.set("YouTube asked for a sign-in.")
        self._refresh_youtube_status()
        if ask_youtube_login(self, failed=True):
            self._refresh_youtube_status()
            self._busy = True
            self.status_var.set("Retrying download…")
            threading.Thread(
                target=self._download_worker,
                args=(source, True, True),
                daemon=True,
            ).start()
            return
        self._on_error(message)

    def _on_video_ready(self, video_path: Path, title: str) -> None:
        self._busy = False
        self._video_path = video_path
        self._title = title
        self._suggested_name = _safe_filename(title) + ".mid"
        self._midi_path = None
        self._recreated_path = None
        self._result = None
        self._detect_range = None
        try:
            self.trim_video.open(video_path)
        except RuntimeError as exc:
            self._on_error(str(exc))
            return
        duration = self.trim_video.duration
        self._syncing = True
        self.trim_seek.configure(from_=0, to=max(duration, 0.01))
        self.start_scale.configure(from_=0, to=max(duration, 0.01))
        self.end_scale.configure(from_=0, to=max(duration, 0.01))
        self.start_var.set(0.0)
        self.end_var.set(duration)
        self._syncing = False
        self._apply_trim_range()
        self.progress["value"] = 100
        self.status_var.set("Set the start and end, then extract MIDI.")
        self._show("trim")

    def _toggle_trim(self) -> None:
        if self.trim_video.playing:
            self.trim_video.pause()
            self.trim_play_btn.config(text="Play")
        else:
            self.trim_video.play()
            self.trim_play_btn.config(text="Pause")

    def _on_trim_time(self, seconds: float) -> None:
        if self._syncing:
            return
        self._syncing = True
        self.trim_seek.set(seconds)
        self._syncing = False
        self.trim_time_var.set(_fmt_time(seconds))
        if not self.trim_video.playing:
            self.trim_play_btn.config(text="Play")

    def _on_trim_seek(self, value: str) -> None:
        if self._syncing or self.trim_video.playing:
            return
        self.trim_play_btn.config(text="Play")
        self.trim_video.seek(float(value))

    def _set_start_here(self) -> None:
        start = min(self.trim_video.time, self.end_var.get() - 0.05)
        self.start_var.set(max(0.0, start))
        self._apply_trim_range()

    def _set_end_here(self) -> None:
        end = max(self.trim_video.time, self.start_var.get() + 0.05)
        self.end_var.set(min(self.trim_video.duration, end))
        self._apply_trim_range()

    def _on_start_moved(self, _value: str) -> None:
        if self._syncing:
            return
        if self.start_var.get() > self.end_var.get() - 0.05:
            self.start_var.set(self.end_var.get() - 0.05)
        self._apply_trim_range()

    def _on_end_moved(self, _value: str) -> None:
        if self._syncing:
            return
        if self.end_var.get() < self.start_var.get() + 0.05:
            self.end_var.set(self.start_var.get() + 0.05)
        self._apply_trim_range()

    def _apply_trim_range(self) -> None:
        start, end = self.start_var.get(), self.end_var.get()
        self.range_var.set(f"Range {_fmt_time(start)} – {_fmt_time(end)}  ({_fmt_time(end - start)})")

    def _trim_next(self) -> None:
        if self._busy or self._video_path is None:
            return
        start, end = self.start_var.get(), self.end_var.get()
        if end <= start:
            messagebox.showinfo("Empty range", "End must be after start.", parent=self)
            return
        if self._detect_range == (start, end) and self._result is not None:
            self._open_compare()
            return
        self.trim_video.pause()
        self._busy = True
        self.progress["value"] = 0
        self.status_var.set("Reading keyboard and notes…")
        threading.Thread(target=self._detect_worker, args=(start, end), daemon=True).start()

    def _detect_worker(self, start: float, end: float) -> None:
        try:
            assert self._video_path is not None
            assert self._tempdir is not None
            result = detect_notes(
                str(self._video_path),
                lambda frac: self._progress_from_thread("detect", frac),
                start_s=start,
                end_s=end,
            )
            midi_path = self._tempdir / "output.mid"
            write_midi(result.notes, midi_path)
            rec_path = self._tempdir / "output.recreated.wav"
            try:
                render_notes_wav(result.notes, rec_path, duration=end - start)
            except Exception:
                rec_path = None
        except (DetectionError, ValueError, OSError) as exc:
            self.after(0, self._on_error, str(exc))
            return
        except Exception as exc:
            self.after(0, self._on_error, f"Unexpected error: {exc}")
            return
        self.after(0, self._on_detect_done, midi_path, rec_path, result, start, end)

    def _on_detect_done(
        self,
        midi_path: Path,
        rec_path: Path | None,
        result: DetectionResult,
        start: float,
        end: float,
    ) -> None:
        self._busy = False
        self._midi_path = midi_path
        self._recreated_path = rec_path
        self._result = result
        self._detect_range = (start, end)
        self.progress["value"] = 100
        self.status_var.set(f"{len(result.notes)} notes, {result.range_label}")
        self._open_compare()

    def _open_compare(self) -> None:
        if self._video_path is None or self._result is None or self._detect_range is None:
            return
        start, end = self._detect_range
        duration = max(0.05, end - start)
        try:
            self.compare_video.open(self._video_path)
        except RuntimeError as exc:
            self._on_error(str(exc))
            return
        self.compare_video.set_range(start, end)
        self.compare_video.set_recreated_audio(self._recreated_path, offset=start)
        self.orig_audio_var.set(True)
        self.rec_audio_var.set(True)
        if self.compare_video.has_original_audio:
            self.orig_audio_btn.state(["!disabled"])
        else:
            self.orig_audio_var.set(False)
            self.orig_audio_btn.state(["disabled"])
        if self._recreated_path is not None:
            self.rec_audio_btn.state(["!disabled"])
        else:
            self.rec_audio_var.set(False)
            self.rec_audio_btn.state(["disabled"])
        self._on_mix_changed()
        self.compare_video.seek(start)
        self.roll.set_notes(
            self._result.notes, self._result.midi_low, self._result.midi_high, duration
        )
        self._syncing = True
        self.compare_seek.configure(from_=0, to=duration)
        self.compare_seek.set(0)
        self._syncing = False
        self.compare_time_var.set(_fmt_time(0))
        self.compare_play_btn.config(text="Play")
        self._show("compare")

    def _compare_back(self) -> None:
        self.compare_video.pause()
        self._show("trim")

    def _on_mix_changed(self) -> None:
        self.compare_video.set_mix(self.orig_audio_var.get(), self.rec_audio_var.get())

    def _toggle_compare(self) -> None:
        if self.compare_video.playing:
            self.compare_video.pause()
            self.compare_play_btn.config(text="Play")
        else:
            self.compare_video.play()
            self.compare_play_btn.config(text="Pause")

    def _on_compare_time(self, seconds: float) -> None:
        if self._detect_range is None:
            return
        start, end = self._detect_range
        local = min(max(0.0, seconds - start), end - start)
        if not self._syncing:
            self._syncing = True
            self.compare_seek.set(local)
            self._syncing = False
        self.compare_time_var.set(_fmt_time(local))
        self.roll.set_playhead(local)
        if not self.compare_video.playing:
            self.compare_play_btn.config(text="Play")

    def _on_compare_seek(self, value: str) -> None:
        if self._syncing or self._detect_range is None or self.compare_video.playing:
            return
        start, _end = self._detect_range
        self.compare_play_btn.config(text="Play")
        self.compare_video.seek(start + float(value))

    def _progress_from_thread(self, phase: str, fraction: float) -> None:
        now = time.monotonic()
        if now - self._last_progress_ui < 0.08 and fraction < 0.999:
            return
        self._last_progress_ui = now
        self.after(0, lambda p=phase, f=fraction: self._set_progress(p, f))

    def _set_progress(self, phase: str, fraction: float) -> None:
        labels = {
            "download": "Downloading video…",
            "detect": "Reading keyboard and notes…",
        }
        self.status_var.set(labels.get(phase, "Working…"))
        self.progress["value"] = max(0.0, min(100.0, fraction * 100.0))

    def _on_error(self, message: str) -> None:
        self._busy = False
        self.progress["value"] = 0
        self.status_var.set(message)
        messagebox.showerror("Error", message, parent=self)

    def _save_midi(self) -> None:
        if self._midi_path is None or not self._midi_path.is_file():
            messagebox.showerror("Nothing to save", "Extract MIDI first.", parent=self)
            return
        dest = filedialog.asksaveasfilename(
            title="Save MIDI",
            defaultextension=".mid",
            initialfile=self._suggested_name,
            filetypes=[("MIDI", "*.mid *.midi"), ("All files", "*.*")],
        )
        if not dest:
            return
        try:
            shutil.copy2(self._midi_path, dest)
        except OSError as exc:
            messagebox.showerror("Save failed", str(exc), parent=self)
            return
        self.status_var.set(f"Saved {Path(dest).name}")

    def _reset_tempdir(self) -> None:
        self._cleanup_temp()
        self._tempdir = Path(tempfile.mkdtemp(prefix="piano-roll-"))

    def _cleanup_temp(self) -> None:
        self.trim_video.close()
        self.compare_video.close()
        if self._tempdir and self._tempdir.exists():
            shutil.rmtree(self._tempdir, ignore_errors=True)
        self._tempdir = None

    def _on_close(self) -> None:
        self._cleanup_temp()
        self.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Piano Roll → MIDI")
    parser.add_argument(
        "--debug",
        nargs="?",
        const="",
        metavar="VIDEO",
        help="OpenCV overlay: keys, sample strips, live hits. Pass a video or URL.",
    )
    args = parser.parse_args()
    if args.debug is not None:
        from debug_view import run_debug

        source = args.debug
        if not source:
            from debug_view import ask_debug_source

            source = ask_debug_source() or ""
            if not source:
                raise SystemExit("No video selected.")
        try:
            run_debug(source)
        except (DetectionError, DownloadError) as exc:
            raise SystemExit(str(exc)) from exc
        return

    app = PianoRollApp()
    app.mainloop()


if __name__ == "__main__":
    main()
