"""Write a single-track piano MIDI from detected notes."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pretty_midi

from detector import Note

DEFAULT_VELOCITY = 80
ACOUSTIC_GRAND = 0


def write_midi(
    notes: Iterable[Note],
    path: str | Path,
    velocity: int = DEFAULT_VELOCITY,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    piano = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=ACOUSTIC_GRAND, name="Piano")
    vel = max(1, min(127, int(velocity)))

    for note in notes:
        start = float(note.start)
        end = float(note.end)
        if end <= start:
            continue
        if not 0 <= note.pitch <= 127:
            continue
        instrument.notes.append(
            pretty_midi.Note(velocity=vel, pitch=int(note.pitch), start=start, end=end)
        )

    if not instrument.notes:
        raise ValueError("No valid notes to write.")

    instrument.notes.sort(key=lambda n: (n.start, n.pitch))
    piano.instruments.append(instrument)
    piano.write(str(path))
    return path
