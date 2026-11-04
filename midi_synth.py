"""Render extracted notes to a simple piano-ish WAV for A/B compare."""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Iterable

import numpy as np
import pretty_midi

from detector import Note
from midi_writer import ACOUSTIC_GRAND, DEFAULT_VELOCITY

PCM_RATE = 44100
PEAK = 0.89


def _piano_wave(phase: np.ndarray) -> np.ndarray:
    return (
        np.sin(phase)
        + 0.45 * np.sin(2.0 * phase)
        + 0.22 * np.sin(3.0 * phase)
        + 0.10 * np.sin(4.0 * phase)
    ) / 1.77


def render_notes_wav(
    notes: Iterable[Note],
    dest: str | Path,
    duration: float,
    velocity: int = DEFAULT_VELOCITY,
    sample_rate: int = PCM_RATE,
) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    n_samples = max(1, int(round(max(duration, 0.05) * sample_rate)))

    piano = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=ACOUSTIC_GRAND, name="Piano")
    vel = max(1, min(127, int(velocity)))
    for note in notes:
        start = float(note.start)
        end = float(note.end)
        if end <= start or not 0 <= note.pitch <= 127:
            continue
        instrument.notes.append(
            pretty_midi.Note(velocity=vel, pitch=int(note.pitch), start=start, end=end)
        )
    if instrument.notes:
        instrument.notes.sort(key=lambda n: (n.start, n.pitch))
        piano.instruments.append(instrument)
        audio = piano.synthesize(fs=sample_rate, wave=_piano_wave)
    else:
        audio = np.zeros(0, dtype=np.float64)

    if audio.size < n_samples:
        audio = np.pad(audio, (0, n_samples - audio.size))
    else:
        audio = audio[:n_samples]
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1e-8:
        audio = audio * (PEAK / peak)
    pcm = np.clip(np.column_stack([audio, audio]) * 32767.0, -32768, 32767).astype(np.int16)

    with wave.open(str(dest), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return dest
