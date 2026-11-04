"""Play a video's soundtrack with ffplay, seeking with the picture."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
from pathlib import Path


def resolve_audio_path(video: Path) -> Path:
    sidecars = sorted(video.parent.glob(f"{video.stem}.audio.*"))
    return sidecars[0] if sidecars else video


def _ffplay() -> str | None:
    return shutil.which("ffplay")


def _atempo_filter(rate: float) -> str:
    remaining = max(0.125, min(8.0, rate))
    parts: list[str] = []
    while remaining > 2.0001:
        parts.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.4999:
        parts.append("atempo=0.5")
        remaining *= 2.0
    parts.append(f"atempo={remaining:.4f}")
    return ",".join(parts)


class FileAudioPlayer:
    """Audio-only sidecar for an OpenCV preview. No-ops if ffplay is missing."""

    def __init__(self) -> None:
        self._path: Path | None = None
        self._proc: subprocess.Popen | None = None

    def open(self, path: Path) -> None:
        self.close()
        self._path = resolve_audio_path(path)

    def play(self, start: float, end: float | None = None, rate: float = 1.0) -> None:
        self.stop()
        exe = _ffplay()
        if exe is None or self._path is None:
            return
        start = max(0.0, start)
        cmd = [
            exe,
            "-nodisp",
            "-vn",
            "-autoexit",
            "-loglevel",
            "quiet",
            "-ss",
            f"{start:.3f}",
        ]
        if end is not None and end > start:
            cmd.extend(["-t", f"{end - start:.3f}"])
        if abs(rate - 1.0) > 1e-3:
            cmd.extend(["-af", _atempo_filter(rate)])
        cmd.append(str(self._path))
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    def stop(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except OSError:
            proc.terminate()
        try:
            proc.wait(timeout=0.4)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                proc.kill()
            proc.wait(timeout=0.4)

    def close(self) -> None:
        self.stop()
        self._path = None
