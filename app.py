#!/usr/bin/env python3
"""Tkinter UI: YouTube URL or local video → MIDI file."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from detector import DetectionError, DetectionResult, detect_notes
from downloader import DownloadError, download_youtube, is_url, validate_local_video
from midi_writer import write_midi


class PianoRollApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Piano Roll → MIDI")
        self.minsize(480, 340)
        self.geometry("520x360")

        self._tempdir: Path | None = None
        self._midi_path: Path | None = None
        self._suggested_name = "output.mid"
        self._busy = False
        self._last_progress_ui = 0.0

        self._build()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build(self) -> None:
        root = ttk.Frame(self, padding=16)
        root.pack(fill=tk.BOTH, expand=True)

        ttk.Label(root, text="Piano Roll → MIDI", font=("TkDefaultFont", 14, "bold")).pack(
            anchor=tk.W
        )
        ttk.Label(
            root,
            text="Turn a Synthesia-style piano tutorial into a MIDI file.",
        ).pack(anchor=tk.W, pady=(0, 12))

        ttk.Label(root, text="YouTube URL").pack(anchor=tk.W)
        self.url_var = tk.StringVar()
        self.url_entry = ttk.Entry(root, textvariable=self.url_var)
        self.url_entry.pack(fill=tk.X, pady=(2, 8))
        self.url_var.trace_add("write", self._on_url_typed)

        ttk.Label(root, text="or").pack(anchor=tk.W)

        file_row = ttk.Frame(root)
        file_row.pack(fill=tk.X, pady=(4, 12))
        ttk.Button(file_row, text="Choose local video…", command=self._choose_video).pack(
            side=tk.LEFT
        )
        self.file_var = tk.StringVar(value="No file selected")
        ttk.Label(file_row, textvariable=self.file_var).pack(
            side=tk.LEFT, padx=(10, 0), fill=tk.X, expand=True
        )
        self._local_path: Path | None = None

        btn_row = ttk.Frame(root)
        btn_row.pack(fill=tk.X, pady=(0, 10))
        self.convert_btn = ttk.Button(btn_row, text="Convert", command=self._start_convert)
        self.convert_btn.pack(side=tk.LEFT)
        self.save_btn = ttk.Button(
            btn_row, text="Save MIDI…", command=self._save_midi, state=tk.DISABLED
        )
        self.save_btn.pack(side=tk.LEFT, padx=(8, 0))

        self.progress = ttk.Progressbar(root, mode="determinate", maximum=100)
        self.progress.pack(fill=tk.X, pady=(4, 6))

        self.status_var = tk.StringVar(value="Waiting for a video")
        ttk.Label(root, textvariable=self.status_var, wraplength=480).pack(anchor=tk.W)

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

    def _start_convert(self) -> None:
        if self._busy:
            return
        url = self.url_var.get().strip()
        if url:
            source = url
            use_url = True
        elif self._local_path is not None:
            source = str(self._local_path)
            use_url = False
        else:
            messagebox.showinfo(
                "No video",
                "Paste a YouTube URL or choose a local video first.",
                parent=self,
            )
            return

        self._busy = True
        self._midi_path = None
        self.convert_btn.config(state=tk.DISABLED)
        self.save_btn.config(state=tk.DISABLED)
        self.progress["value"] = 0
        self.status_var.set("Starting…")
        self._reset_tempdir()

        thread = threading.Thread(
            target=self._worker,
            args=(source, use_url),
            daemon=True,
        )
        thread.start()

    def _worker(self, source: str, use_url: bool) -> None:
        try:
            assert self._tempdir is not None
            midi_path, title, result = _run_conversion(
                source,
                use_url,
                self._tempdir,
                self._progress_from_thread,
            )
        except (DownloadError, DetectionError, ValueError, OSError) as exc:
            self.after(0, self._on_error, str(exc))
            return
        except Exception as exc:
            self.after(0, self._on_error, f"Unexpected error: {exc}")
            return
        self.after(0, self._on_done, midi_path, title, result)

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

    def _on_done(self, midi_path: Path, title: str, result: DetectionResult) -> None:
        self._busy = False
        self._midi_path = midi_path
        self._suggested_name = _safe_filename(title) + ".mid"
        self.convert_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.NORMAL)
        self.progress["value"] = 100
        self.status_var.set(
            f"Done — {len(result.notes)} notes, {result.range_label}. Save the MIDI file."
        )

    def _on_error(self, message: str) -> None:
        self._busy = False
        self.convert_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.DISABLED)
        self.progress["value"] = 0
        self.status_var.set(message)

    def _save_midi(self) -> None:
        if self._midi_path is None or not self._midi_path.is_file():
            messagebox.showerror("Nothing to save", "Convert a video first.", parent=self)
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
        if self._tempdir and self._tempdir.exists():
            shutil.rmtree(self._tempdir, ignore_errors=True)
        self._tempdir = None

    def _on_close(self) -> None:
        self._cleanup_temp()
        self.destroy()


def _run_conversion(
    source: str,
    use_url: bool,
    tmpdir: Path,
    progress,
) -> tuple[Path, str, DetectionResult]:
    if use_url:
        if not is_url(source):
            raise DownloadError("Enter a valid http(s) YouTube URL.")
        video_path, title = download_youtube(
            source, tmpdir, lambda frac: progress("download", frac)
        )
    else:
        video_path = validate_local_video(source)
        title = video_path.stem

    progress("detect", 0.0)
    result = detect_notes(str(video_path), lambda frac: progress("detect", frac))
    midi_path = tmpdir / "output.mid"
    write_midi(result.notes, midi_path)
    return midi_path, title, result


def _safe_filename(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in " -_." else "_" for ch in name)
    cleaned = cleaned.strip(" ._") or "output"
    return cleaned[:80]


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
