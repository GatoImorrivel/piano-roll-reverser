"""Play a video's soundtrack, pause/resume in place, and seek by sample."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path

PCM_RATE = 44100
PCM_CHANNELS = 2
PCM_WIDTH = 2
PCM_BPS = PCM_RATE * PCM_CHANNELS * PCM_WIDTH
PAPLAY_LATENCY = 0.06


def resolve_audio_path(video: Path) -> Path:
    sidecars = sorted(video.parent.glob(f"{video.stem}.audio.*"))
    return sidecars[0] if sidecars else video


def _ffmpeg() -> str | None:
    return shutil.which("ffmpeg")


def _paplay() -> str | None:
    return shutil.which("paplay")


def _ffplay() -> str | None:
    return shutil.which("ffplay")


def _default_wav_path(video: Path) -> Path:
    parent = video.parent.name
    if parent.startswith("piano-roll"):
        return video.with_name(video.stem + ".pcm.wav")
    return Path(tempfile.gettempdir()) / f"piano-roll-{video.stem}.pcm.wav"


def ensure_pcm_wav(video: Path, dest: Path | None = None) -> Path | None:
    """Decode the soundtrack to a seekable WAV. None if the file is silent."""
    exe = _ffmpeg()
    if exe is None:
        return None
    dest = dest or _default_wav_path(video)
    src = resolve_audio_path(video)
    if dest.is_file() and dest.stat().st_size > 64:
        try:
            if dest.stat().st_mtime >= src.stat().st_mtime:
                return dest
        except OSError:
            return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            exe,
            "-y",
            "-i",
            str(src),
            "-vn",
            "-ac",
            str(PCM_CHANNELS),
            "-ar",
            str(PCM_RATE),
            "-c:a",
            "pcm_s16le",
            str(dest),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0 or not dest.is_file() or dest.stat().st_size <= 64:
        try:
            dest.unlink()
        except OSError:
            pass
        return None
    return dest


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
    """Pause freezes the stream; seek restarts from an exact WAV offset."""

    def __init__(self) -> None:
        self._wav: Path | None = None
        self._proc: subprocess.Popen | None = None
        self._helper: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._paused = False
        self._start_sec = 0.0
        self._end: float | None = None
        self._media_rate = 1.0
        self._bytes_sent = 0
        self._frozen_pos = 0.0
        self._play_mono = 0.0
        self._startup_delay = 0.0
        self._use_wall = False

    def open(self, path: Path) -> None:
        self.close()
        self._wav = ensure_pcm_wav(path)

    def play(self, start: float, end: float | None = None, rate: float = 1.0) -> None:
        self.stop()
        if self._wav is None:
            return
        self._start_sec = max(0.0, start)
        self._end = end
        self._media_rate = rate if rate > 1e-3 else 1.0
        self._bytes_sent = 0
        self._paused = False
        self._frozen_pos = self._start_sec
        self._play_mono = time.monotonic()
        self._startup_delay = PAPLAY_LATENCY
        self._stop_evt = threading.Event()
        self._use_wall = abs(self._media_rate - 1.0) > 1e-3
        if self._use_wall:
            self._start_rated()
        else:
            self._start_pcm()

    def pause(self) -> None:
        with self._lock:
            if self._paused or self._proc is None or self._proc.poll() is not None:
                return
            self._frozen_pos = self._heard_unlocked()
            self._signal(signal.SIGSTOP)
            self._paused = True

    def resume(self) -> bool:
        with self._lock:
            if not self._paused or self._proc is None or self._proc.poll() is not None:
                return False
            self._start_sec = self._frozen_pos
            self._play_mono = time.monotonic()
            self._startup_delay = 0.0
            self._signal(signal.SIGCONT)
            self._paused = False
            return True

    def stop(self) -> None:
        self._stop_evt.set()
        proc = self._proc
        helper = self._helper
        thread = self._thread
        self._proc = None
        self._helper = None
        self._thread = None
        self._paused = False
        if proc is not None:
            self._kill(proc, group=True)
        if helper is not None:
            self._kill(helper, group=False)
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=0.6)

    def close(self) -> None:
        self.stop()
        self._wav = None

    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None and not self._paused

    def position(self) -> float:
        with self._lock:
            if self._paused:
                return self._frozen_pos
            return self._heard_unlocked()

    def _heard_unlocked(self) -> float:
        elapsed = 0.0 if self._paused else (time.monotonic() - self._play_mono)
        heard = max(0.0, elapsed - self._startup_delay)
        return self._start_sec + heard * self._media_rate

    def _start_pcm(self) -> None:
        sink = self._open_sink()
        if sink is None:
            return
        self._proc = sink
        self._thread = threading.Thread(target=self._feed, daemon=True)
        self._thread.start()

    def _start_rated(self) -> None:
        ffmpeg = _ffmpeg()
        if ffmpeg is None or self._wav is None:
            return
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-ss", f"{self._start_sec:.4f}"]
        if self._end is not None and self._end > self._start_sec:
            cmd.extend(["-t", f"{self._end - self._start_sec:.4f}"])
        cmd.extend(
            [
                "-i",
                str(self._wav),
                "-af",
                _atempo_filter(self._media_rate),
                "-f",
                "s16le",
                "-ac",
                str(PCM_CHANNELS),
                "-ar",
                str(PCM_RATE),
                "pipe:1",
            ]
        )
        helper = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        assert helper.stdout is not None
        sink = self._open_sink(stdin=helper.stdout)
        helper.stdout.close()
        if sink is None:
            self._kill(helper, group=False)
            return
        self._helper = helper
        self._proc = sink
        self._use_wall = True

    def _open_sink(self, stdin=subprocess.PIPE) -> subprocess.Popen | None:
        paplay = _paplay()
        if paplay is not None:
            return subprocess.Popen(
                [
                    paplay,
                    "--raw",
                    f"--rate={PCM_RATE}",
                    f"--channels={PCM_CHANNELS}",
                    "--format=s16le",
                    f"--latency-msec={int(PAPLAY_LATENCY * 1000)}",
                ],
                stdin=stdin,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        exe = _ffplay()
        if exe is None:
            return None
        self._use_wall = True
        return subprocess.Popen(
            [
                exe,
                "-nodisp",
                "-autoexit",
                "-loglevel",
                "quiet",
                "-f",
                "s16le",
                "-ar",
                str(PCM_RATE),
                "-ac",
                str(PCM_CHANNELS),
                "-i",
                "pipe:0",
            ],
            stdin=stdin,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    def _feed(self) -> None:
        if self._wav is None:
            return
        proc = self._proc
        if proc is None or proc.stdin is None:
            return
        try:
            with wave.open(str(self._wav), "rb") as wav:
                frame_rate = max(wav.getframerate(), 1)
                start_frame = int(round(self._start_sec * frame_rate))
                wav.setpos(min(max(0, start_frame), wav.getnframes()))
                end_bytes = None
                if self._end is not None and self._end > self._start_sec:
                    end_bytes = int((self._end - self._start_sec) * PCM_BPS)
                while not self._stop_evt.is_set():
                    frames = 2048
                    if end_bytes is not None:
                        left = end_bytes - self._bytes_sent
                        if left <= 0:
                            break
                        frames = min(frames, max(1, left // (PCM_CHANNELS * PCM_WIDTH)))
                    data = wav.readframes(frames)
                    if not data:
                        break
                    try:
                        proc.stdin.write(data)
                    except BrokenPipeError:
                        break
                    except ValueError:
                        break
                    with self._lock:
                        self._bytes_sent += len(data)
        finally:
            try:
                proc.stdin.close()
            except Exception:
                pass

    def _signal(self, sig: int) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        try:
            os.killpg(proc.pid, sig)
        except OSError:
            try:
                proc.send_signal(sig)
            except OSError:
                pass

    def _kill(self, proc: subprocess.Popen, group: bool = True) -> None:
        if proc.poll() is not None:
            return
        if group:
            try:
                os.killpg(proc.pid, signal.SIGCONT)
            except OSError:
                pass
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except OSError:
                proc.terminate()
        else:
            proc.terminate()
        try:
            proc.wait(timeout=0.4)
        except subprocess.TimeoutExpired:
            if group:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    proc.kill()
            else:
                proc.kill()
            try:
                proc.wait(timeout=0.4)
            except subprocess.TimeoutExpired:
                pass
