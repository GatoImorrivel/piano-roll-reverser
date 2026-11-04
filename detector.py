"""Locate a Synthesia-style keyboard and turn key lighting into notes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

ProgressFn = Callable[[float], None]

MIN_NOTE_SEC = 0.03
DEBOUNCE_FRAMES = 2
SAT_ON = 45.0
SAT_ON_BLACK = 58.0
SAT_DELTA = 25.0
SAT_DELTA_BLACK = 32.0
VAL_DELTA_WHITE = 40.0
VAL_ON_BLACK = 40.0
VAL_DELTA_BLACK = 28.0
MIN_VAL_FOR_SAT = 26.0
BLEED_RATIO = 0.92

# white-key count → (lowest MIDI, highest MIDI)
KEYBOARD_RANGES = {
    52: (21, 108),  # 88-key A0–C8
    45: (28, 103),  # 76-key E1–G7
    36: (36, 96),  # 61-key C2–C7
    29: (36, 84),  # 49-key C2–C6
}

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


class DetectionError(Exception):
    """Raised when the keyboard or notes cannot be recovered."""


@dataclass(frozen=True)
class Note:
    pitch: int
    start: float
    end: float
    strength: float = 0.0


@dataclass
class Key:
    midi: int
    x0: int
    x1: int
    y0: int
    y1: int
    is_black: bool
    mid_x: int = 0
    baseline_sat: float = 0.0
    baseline_val: float = 0.0


@dataclass
class DetectionResult:
    notes: list[Note]
    n_white: int
    midi_low: int
    midi_high: int
    fps: float
    frame_count: int

    @property
    def range_label(self) -> str:
        span = self.midi_high - self.midi_low + 1
        return f"{span}-key ({midi_name(self.midi_low)}–{midi_name(self.midi_high)})"


def midi_name(pitch: int) -> str:
    return f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def is_black_key(midi: int) -> bool:
    return (midi % 12) in (1, 3, 6, 8, 10)


def detect_notes(video_path: str, progress_cb: ProgressFn | None = None) -> DetectionResult:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise DetectionError("Could not open the video file.")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 1e-3:
            fps = 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        _report(progress_cb, 0.02)
        frame = _calibration_frame(cap, frame_count)
        keys = _build_keyboard(frame)
        _report(progress_cb, 0.08)

        _calibrate_baselines(cap, keys, frame_count)
        _report(progress_cb, 0.16)

        notes = _scan_notes(cap, keys, fps, frame_count, progress_cb)
    finally:
        cap.release()

    if not notes:
        raise DetectionError(
            "Keyboard found, but no notes were detected. "
            "The video skin may not be supported."
        )

    midi_low = min(k.midi for k in keys)
    midi_high = max(k.midi for k in keys)
    n_white = sum(1 for k in keys if not k.is_black)
    return DetectionResult(
        notes=notes,
        n_white=n_white,
        midi_low=midi_low,
        midi_high=midi_high,
        fps=fps,
        frame_count=frame_count,
    )


def _report(progress_cb: ProgressFn | None, fraction: float) -> None:
    if progress_cb:
        progress_cb(min(1.0, max(0.0, fraction)))


def _calibration_frame(cap: cv2.VideoCapture, frame_count: int) -> np.ndarray:
    """Prefer a quiet intro frame; mid-video falling notes split the keyboard."""
    candidates = [0]
    if frame_count > 12:
        candidates.append(12)
    if frame_count > 40:
        candidates.extend(int(frame_count * frac) for frac in (0.04, 0.10, 0.20, 0.30))
    best_frame: np.ndarray | None = None
    best_score = -1e9
    for index in dict.fromkeys(candidates):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        score = _layout_quality(frame)
        if score > best_score:
            best_score = score
            best_frame = frame
    if best_frame is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise DetectionError("Could not read frames from the video.")
        return frame
    return best_frame


def _layout_quality(frame: np.ndarray) -> float:
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        smooth = cv2.GaussianBlur(gray, (3, 3), 0)
        y0, y1 = _find_keyboard_band(smooth)
        y0, y1 = _refine_white_apron(gray, y0, y1)
        y0, y1 = _extend_keyboard_bottom(gray, y0, y1)
        y0, y1 = _extend_keyboard_top(gray, y0, y1)
        x0, x1 = _find_keyboard_x(gray, y0, y1)
        centers = _detect_black_centers(gray, x0, x1, y0, y1)
    except DetectionError:
        return -1.0
    width_frac = (x1 - x0) / max(frame.shape[1], 1)
    return float(len(centers) + 6 * len(_f_sharp_indices(centers)) + 12 * width_frac)


def _build_keyboard(frame: np.ndarray) -> list[Key]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    y0, y1 = _find_keyboard_band(smooth)
    y0, y1 = _refine_white_apron(gray, y0, y1)
    y0, y1 = _extend_keyboard_bottom(gray, y0, y1)
    y0, y1 = _extend_keyboard_top(gray, y0, y1)
    x0, x1 = _find_keyboard_x(gray, y0, y1)
    centers = _detect_black_centers(gray, x0, x1, y0, y1)
    n_white = _count_white_keys(gray, x0, x1, y0, y1)
    n_white = _snap_white_count(n_white, x0, x1, frame.shape[1], len(centers))
    if n_white in KEYBOARD_RANGES:
        x0, x1 = _phase_align_keyboard(gray, x0, x1, y0, y1, n_white)
    keyed = _keys_from_black_lattice(gray, x0, x1, y0, y1)
    if keyed:
        return keyed
    extra_top = max(0, y0 - (y1 - y0))
    if extra_top < y0:
        keyed = _keys_from_black_lattice(gray, x0, x1, extra_top, y1)
        if keyed:
            return keyed
    if n_white in KEYBOARD_RANGES:
        return _geometric_keys(n_white, x0, x1, y0, y1)
    snapped = min(KEYBOARD_RANGES, key=lambda standard: abs(standard - max(n_white, 29)))
    return _geometric_keys(snapped, x0, x1, y0, y1)


def _extend_keyboard_bottom(gray: np.ndarray, y0: int, y1: int) -> tuple[int, int]:
    """Grow the band through the white-key apron under the black keys."""
    h = gray.shape[0]
    floor = max(float(gray[y0:y1].std()) * 0.20, 6.0)
    last = y1
    weak = 0
    for i in range(y1, h):
        if float(gray[i].std()) >= floor:
            last = i
            weak = 0
        else:
            weak += 1
            if weak >= 3:
                break
    return y0, min(h - 1, last)


def _refine_white_apron(gray: np.ndarray, y0: int, y1: int) -> tuple[int, int]:
    """If falling notes merged into the band, keep the bright white-key strip."""
    if y1 - y0 < 16:
        return y0, y1
    means = gray[y0:y1].mean(axis=1)
    tail = means[int(len(means) * 0.60) :]
    if tail.size == 0:
        return y0, y1
    floor = float(np.median(tail)) * 0.78
    top = 0
    weak = 0
    for i in range(len(means) - 1, -1, -1):
        if means[i] >= floor:
            top = i
            weak = 0
        else:
            weak += 1
            if weak >= 6:
                break
    return y0 + top, y1


def _extend_keyboard_top(gray: np.ndarray, y0: int, y1: int) -> tuple[int, int]:
    """Grow upward through the black-key zone above the white apron."""
    height = max(1, y1 - y0)
    max_up = min(y0, max(int(height * 2.4), int(gray.shape[0] * 0.35)))
    if max_up < 4:
        return y0, y1
    min_score = max(_row_keyboard_score(gray[min(y0, gray.shape[0] - 1)]), 30.0) * 0.22
    last = y0
    weak = 0
    for i in range(y0 - 1, y0 - max_up - 1, -1):
        if _row_keyboard_score(gray[i]) >= min_score:
            last = i
            weak = 0
        else:
            weak += 1
            if weak >= 5:
                break
    return last, y1


def _row_keyboard_score(row: np.ndarray) -> float:
    std = float(row.std())
    if std < 5.0:
        return 0.0
    binary = row > float(np.median(row))
    transitions = int(np.count_nonzero(np.diff(binary.astype(np.int8))))
    if transitions < 8:
        return 0.0
    return std * min(transitions, 140)


def _true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, bit in enumerate(np.append(mask.astype(bool), False)):
        if bit and start is None:
            start = i
        elif not bit and start is not None:
            runs.append((start, i))
            start = None
    return runs


def _find_keyboard_band(gray: np.ndarray) -> tuple[int, int]:
    h, _w = gray.shape
    search_top = int(h * 0.08)
    region = gray[search_top:, :]
    if region.size == 0:
        raise DetectionError("Could not detect a piano keyboard.")

    scores = np.array([_row_keyboard_score(row) for row in region], dtype=np.float32)
    peak = float(scores.max()) if scores.size else 0.0
    if peak < 40.0:
        raise DetectionError("Could not detect a piano keyboard.")

    mask = scores >= max(peak * 0.40, 40.0)
    min_h = max(12, int(h * 0.06))
    runs = [run for run in _true_runs(mask) if run[1] - run[0] >= min_h]
    if not runs:
        runs = _true_runs(mask)
    if not runs:
        raise DetectionError("Could not detect a piano keyboard.")

    # The keyboard sits at the bottom; falling notes may also score above it.
    start, end = max(runs, key=lambda run: (run[0], run[1] - run[0]))
    y0 = search_top + start
    y1 = min(h - 1, search_top + end)
    if y1 - y0 < 12:
        raise DetectionError("Could not detect a piano keyboard.")

    # Tall run: keep the bottom (apron). extend_top recovers the black keys.
    max_h = int(h * 0.45)
    if y1 - y0 > max_h:
        y0 = y1 - max_h
    return y0, y1


def _find_keyboard_x(gray: np.ndarray, y0: int, y1: int) -> tuple[int, int]:
    height = y1 - y0
    lower_top = y0 + int(height * 0.70)
    strip = gray[lower_top:y1, :] if height > 8 else gray[y0:y1, :]
    score = strip.std(axis=0) + 0.2 * strip.mean(axis=0)
    threshold = max(float(np.percentile(score, 30)), float(score.max()) * 0.16)
    active = score >= threshold
    if not np.any(active):
        return 0, gray.shape[1] - 1

    width = gray.shape[1]
    # Close gaps between white keys so we keep the whole keyboard, not one run.
    # Lit keys can open a hole; close more aggressively than a single white key.
    kernel = np.ones((1, max(21, width // 14)), dtype=np.uint8)
    closed = cv2.morphologyEx(
        active.astype(np.uint8).reshape(1, -1), cv2.MORPH_CLOSE, kernel
    )
    x0, x1 = _longest_true_run(closed.ravel().astype(bool))
    col_mean = strip.mean(axis=0)
    bright = col_mean > max(float(np.percentile(col_mean, 20)), 40.0)
    if float(bright.mean()) > 0.70 and (x1 - x0) < width * 0.82:
        return 0, width - 1
    if x1 - x0 < width * 0.55:
        return 0, width - 1
    return x0, x1


def _phase_align_keyboard(
    gray: np.ndarray, x0: int, x1: int, y0: int, y1: int, n_white: int
) -> tuple[int, int]:
    """Shift the even white-key grid so black keys land on the 2-3-2-3 pattern."""
    midi_low, midi_high = KEYBOARD_RANGES[n_white]
    whites = [m for m in range(midi_low, midi_high + 1) if not is_black_key(m)]
    width = x1 - x0
    if width < 32:
        return x0, x1
    white_w = width / n_white
    height = y1 - y0
    upper = gray[y0 + int(height * 0.10) : y0 + int(height * 0.48), x0:x1]
    if upper.size == 0:
        return x0, x1

    profile = upper.mean(axis=0).astype(np.float64)
    profile = (profile - profile.mean()) / (profile.std() + 1e-6)
    black_offsets = [
        (i + 1) * white_w
        for i in range(len(whites) - 1)
        if whites[i + 1] - whites[i] == 2
    ]
    if not black_offsets:
        return x0, x1

    radius = max(2, int(white_w * 0.12))
    best_shift, best = 0, -1e9
    for shift in range(-int(white_w * 0.55), int(white_w * 0.55) + 1):
        score = 0.0
        for offset in black_offsets:
            center = int(round(offset + shift))
            a = max(0, center - radius)
            b = min(len(profile), center + radius + 1)
            if b > a:
                score -= float(profile[a:b].mean())
        if score > best:
            best, best_shift = score, shift

    new_x0 = x0 + best_shift
    new_x1 = x1 + best_shift
    if new_x0 < 0:
        new_x1 -= new_x0
        new_x0 = 0
    overflow = new_x1 - (gray.shape[1] - 1)
    if overflow > 0:
        new_x0 -= overflow
        new_x1 -= overflow
    return max(0, new_x0), min(gray.shape[1] - 1, new_x1)


def _longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    best_start, best_len, start, length = 0, 0, 0, 0
    padded = np.concatenate([mask.astype(bool).ravel(), [False]])
    for i, bit in enumerate(padded):
        if bit:
            if length == 0:
                start = i
            length += 1
        else:
            if length > best_len:
                best_start, best_len = start, length
            length = 0
    return best_start, best_start + max(best_len - 1, 0)


def _count_white_keys(gray: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> int:
    height = y1 - y0
    lower_top = y0 + int(height * 0.72)
    lower = gray[lower_top:y1, x0:x1]
    if lower.size == 0 or lower.shape[1] < 32:
        return 0

    profile = lower.mean(axis=0).astype(np.float64)
    profile -= profile.mean()
    if float(np.max(np.abs(profile))) < 1.0:
        return 0

    windowed = profile * np.hanning(len(profile))
    spectrum = np.abs(np.fft.rfft(windowed))
    spectrum[0] = 0
    lo, hi = 20, 70
    hi = min(hi, len(spectrum) - 1)
    if hi <= lo:
        return 0
    return lo + int(np.argmax(spectrum[lo : hi + 1]))


def _snap_white_count(
    n_white: int, x0: int, x1: int, frame_w: int, n_black: int = 0
) -> int:
    if n_black >= 8:
        from_black = int(round(n_black * 52 / 36))
        # FFT often reports the 2x harmonic of a close-up keyboard.
        if n_white >= from_black * 1.6:
            n_white = from_black
        elif abs(n_white - from_black) > 10:
            n_white = from_black
    wide = (x1 - x0) / max(frame_w, 1) > 0.80
    if n_white < 20:
        if wide and n_black >= 30:
            return 52
        if wide:
            return max(n_white, 29)
        raise DetectionError("Could not detect piano keys in the keyboard.")

    if wide and 48 <= n_white <= 56 and n_black >= 30:
        return 52

    if n_black >= 8 and abs(n_white - int(round(n_black * 52 / 36))) <= 2:
        return n_white

    snapped = min(KEYBOARD_RANGES, key=lambda standard: abs(standard - n_white))
    if abs(snapped - n_white) > 8:
        raise DetectionError("Could not detect piano keys in the keyboard.")
    return snapped


def _geometric_keys(n_white: int, x0: int, x1: int, y0: int, y1: int) -> list[Key]:
    midi_low, midi_high = KEYBOARD_RANGES[n_white]
    whites = [m for m in range(midi_low, midi_high + 1) if not is_black_key(m)]
    if len(whites) != n_white:
        raise DetectionError("Internal keyboard map mismatch.")

    width = x1 - x0
    white_w = width / n_white
    height = y1 - y0
    keys: list[Key] = []

    for i, midi in enumerate(whites):
        xa = int(round(x0 + i * white_w))
        xb = int(round(x0 + (i + 1) * white_w))
        keys.append(
            Key(
                midi=midi,
                x0=xa,
                x1=xb,
                y0=y0,
                y1=y1,
                is_black=False,
                mid_x=(xa + xb) // 2,
            )
        )

    black_h = y0 + int(height * 0.62)
    for i in range(len(whites) - 1):
        if whites[i + 1] - whites[i] != 2:
            continue
        center = x0 + (i + 1) * white_w
        half = white_w * 0.20
        xa = int(round(center - half))
        xb = int(round(center + half))
        keys.append(
            Key(
                midi=whites[i] + 1,
                x0=xa,
                x1=xb,
                y0=y0,
                y1=black_h,
                is_black=True,
                mid_x=int(round(center)),
            )
        )

    keys.sort(key=lambda k: k.midi)
    return keys


# White-key units from C0: C=0, C#=0.5, D=1, ... B=6, then +7 per octave.
_WHITE_COORD = {
    0: 0.0,
    1: 0.5,
    2: 1.0,
    3: 1.5,
    4: 2.0,
    5: 3.0,
    6: 3.5,
    7: 4.0,
    8: 4.5,
    9: 5.0,
    10: 5.5,
    11: 6.0,
}


def _white_coord(midi: int) -> float:
    return (midi // 12) * 7.0 + _WHITE_COORD[midi % 12]


def _detect_black_centers(gray: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> list[float]:
    """Centers of black keys. Ignores printed letters (dark on only a few rows)."""
    height = y1 - y0
    upper = gray[y0 + int(height * 0.06) : y0 + int(height * 0.52), x0:x1]
    if upper.size == 0 or upper.shape[0] < 6:
        return []

    dark_ref = float(np.percentile(upper, 12))
    bright_ref = float(np.percentile(upper, 72))
    mid = (dark_ref + bright_ref) * 0.5
    if bright_ref - dark_ref < 8:
        profile = upper.mean(axis=0).astype(np.float64)
        mid = float(np.percentile(profile, 38))
        dark = profile < mid
    else:
        binary = (upper < mid).astype(np.uint8) * 255
        kernel_h = max(5, upper.shape[0] // 7)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((kernel_h, 2), np.uint8))
        dark_frac = cleaned.mean(axis=0) / 255.0
        dark = dark_frac > 0.40
    runs = _true_runs(dark)
    if not runs:
        return []
    widths = [end - start for start, end in runs]
    med_w = float(np.median(widths))
    # Black keys are ~0.4 of a white key; don't assume an 88-key span.
    lo, hi = max(3.0, med_w * 0.45), med_w * 1.85
    centers: list[float] = []
    for start, end in runs:
        width = end - start
        if lo <= width <= hi:
            centers.append(x0 + (start + end - 1) / 2.0)
    return centers


def _f_sharp_indices(centers: list[float]) -> list[int]:
    """Indices that start a 3-group (F# G# A#)."""
    if len(centers) < 4:
        return []
    gaps = np.diff(centers)
    median = float(np.median(gaps))
    if median <= 0:
        return []
    large = median * 1.35
    found: list[int] = []
    for i in range(len(gaps) - 3):
        if (
            gaps[i] > large
            and gaps[i + 1] <= large
            and gaps[i + 2] <= large
            and gaps[i + 3] > large
        ):
            found.append(i + 1)
    if not found:
        for i in range(len(gaps) - 2):
            if gaps[i] <= large and gaps[i + 1] <= large and gaps[i + 2] > large:
                found.append(i)
    return found


def _f_sharp_index(centers: list[float]) -> int | None:
    found = _f_sharp_indices(centers)
    return found[len(found) // 2] if found else None


_BLACK_PC = (1, 3, 6, 8, 10)  # C# D# F# G# A#


def _black_rel_from_fs(offset: int) -> float:
    """White-key units from F# to the offset-th black key (0 = F#)."""
    index = 2 + offset
    octave, rem = divmod(index, 5)
    return octave * 7.0 + _WHITE_COORD[_BLACK_PC[rem]] - _WHITE_COORD[6]


def _fit_spacing(centers: list[float], fs_idx: int) -> tuple[float, float, float] | None:
    """Least-squares white-key width and F# x from a 2-3-2-3 assignment."""
    rels = np.array([_black_rel_from_fs(i - fs_idx) for i in range(len(centers))])
    xs = np.array(centers, dtype=np.float64)
    rel_m = float(rels.mean())
    xs_m = float(xs.mean())
    var = float(np.sum((rels - rel_m) ** 2))
    if var < 1e-6:
        return None
    white_w = float(np.sum((rels - rel_m) * (xs - xs_m)) / var)
    if white_w < 4:
        return None
    fs_x = xs_m - white_w * rel_m
    pred = fs_x + rels * white_w
    rmse = float(np.sqrt(np.mean((xs - pred) ** 2)))
    return fs_x, white_w, rmse


def _pick_fsharp_octave(fs_x: float, white_w: float, x0: int, x1: int) -> int:
    best_midi = 54
    best = 1e9
    for fs_midi in (30, 42, 54, 66, 78, 90):
        vis = [
            midi
            for midi in range(21, 109)
            if x0 - white_w * 0.3
            <= fs_x + (_white_coord(midi) - _white_coord(fs_midi)) * white_w
            <= x1 + white_w * 0.3
        ]
        if len(vis) < 12:
            continue
        mid = 0.5 * (vis[0] + vis[-1])
        penalty = abs(mid - 60.0)
        if penalty < best:
            best = penalty
            best_midi = fs_midi
    return best_midi


def _fit_black_lattice(
    centers: list[float], x0: int, x1: int
) -> tuple[float, int, float, dict[int, float]] | None:
    anchors = _f_sharp_indices(centers)
    if not anchors:
        return None
    best: tuple[float, int, float, dict[int, float]] | None = None
    best_score = 1e9
    for fs_idx in anchors:
        fitted = _fit_spacing(centers, fs_idx)
        if fitted is None:
            continue
        fs_x, white_w, rmse = fitted
        if rmse > white_w * 0.22:
            continue
        fs_midi = _pick_fsharp_octave(fs_x, white_w, x0, x1)
        assigned: dict[int, float] = {}
        unused = list(centers)
        for midi in range(21, 109):
            if not is_black_key(midi):
                continue
            exp = fs_x + (_white_coord(midi) - _white_coord(fs_midi)) * white_w
            if not unused:
                break
            nearest = min(range(len(unused)), key=lambda i: abs(unused[i] - exp))
            if abs(unused[nearest] - exp) <= white_w * 0.40:
                assigned[midi] = unused.pop(nearest)
        score = rmse - 0.05 * len(assigned)
        if score < best_score:
            best_score = score
            best = (fs_x, fs_midi, white_w, assigned)
    return best


def _keys_from_black_lattice(
    gray: np.ndarray, x0: int, x1: int, y0: int, y1: int
) -> list[Key] | None:
    """Label keys from the 2-3-2-3 black pattern, anchored on F#."""
    centers = _detect_black_centers(gray, x0, x1, y0, y1)
    fitted = _fit_black_lattice(centers, x0, x1)
    if fitted is None:
        return None
    fs_x, fs_midi, white_w, black_xs = fitted
    height = y1 - y0
    black_h = y0 + int(height * 0.62)
    left_bound = min(centers) - white_w * 0.95
    right_bound = max(centers) + white_w * 1.75
    keys: list[Key] = []
    for midi in range(21, 109):
        mid = black_xs.get(
            midi, fs_x + (_white_coord(midi) - _white_coord(fs_midi)) * white_w
        )
        if mid < max(x0 - white_w * 0.2, left_bound) or mid > min(
            x1 + white_w * 0.2, right_bound
        ):
            continue
        black = is_black_key(midi)
        if (
            black
            and midi not in black_xs
            and (mid < min(centers) - white_w * 0.45 or mid > max(centers) + white_w * 0.45)
        ):
            continue
        half = white_w * (0.20 if black else 0.48)
        keys.append(
            Key(
                midi=midi,
                x0=int(round(mid - half)),
                x1=int(round(mid + half)),
                y0=y0,
                y1=black_h if black else y1,
                is_black=black,
                mid_x=int(round(mid)),
            )
        )
    if len(keys) < 20:
        return None
    keys.sort(key=lambda key: key.midi)
    return keys


def _calibrate_baselines(cap: cv2.VideoCapture, keys: list[Key], frame_count: int) -> None:
    samples_sat = [[] for _ in keys]
    samples_val = [[] for _ in keys]

    if frame_count > 40:
        indices = np.linspace(int(frame_count * 0.12), int(frame_count * 0.88), 18)
        for idx in indices.astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            for i, key in enumerate(keys):
                sat, val = _sample_key(hsv, key)
                samples_sat[i].append(sat)
                samples_val[i].append(val)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(12):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            for i, key in enumerate(keys):
                sat, val = _sample_key(hsv, key)
                samples_sat[i].append(sat)
                samples_val[i].append(val)

    for i, key in enumerate(keys):
        if samples_sat[i]:
            key.baseline_sat = float(np.percentile(samples_sat[i], 20))
            key.baseline_val = float(np.percentile(samples_val[i], 20 if key.is_black else 50))
        else:
            key.baseline_sat = 8.0
            key.baseline_val = 40.0 if key.is_black else 200.0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


def _sample_rect(key: Key, width: int, height: int) -> tuple[int, int, int, int]:
    """Pixel box sampled on the key surface: (x0, y0, x1, y1)."""
    cx = key.mid_x or (key.x0 + key.x1) // 2
    if key.is_black:
        cy = int(key.y0 + 0.50 * (key.y1 - key.y0))
        rh = max(2, (key.y1 - key.y0) // 8)
        rw = 1
    else:
        # Above the printed letter row on labeled tutorials.
        cy = int(key.y0 + 0.72 * (key.y1 - key.y0))
        rh = max(2, (key.y1 - key.y0) // 14)
        rw = 1
    xa = max(0, cx - rw)
    xb = min(width, cx + rw + 1)
    ya = max(0, cy - rh)
    yb = min(height, cy + rh + 1)
    return xa, ya, xb, yb


def _sample_key(hsv: np.ndarray, key: Key) -> tuple[float, float]:
    """Thin vertical strip down the middle of the key, on the key surface."""
    h, w = hsv.shape[:2]
    xa, ya, xb, yb = _sample_rect(key, w, h)
    roi = hsv[ya:yb, xa:xb]
    if roi.size == 0:
        return 0.0, 0.0
    sat = float(roi[:, :, 1].mean())
    val = float(roi[:, :, 2].mean())
    # Near-black chroma noise reports sat=255 with val≈0.
    if val < MIN_VAL_FOR_SAT:
        sat = 0.0
    return sat, val


def frame_key_states(frame: np.ndarray, keys: list[Key]) -> tuple[list[bool], list[float]]:
    """Same on/off decision the scanner uses for one frame."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    raw_on: list[bool] = []
    strength: list[float] = []
    for key in keys:
        sat, val = _sample_key(hsv, key)
        on, amount = _key_activation(key, sat, val)
        raw_on.append(on)
        strength.append(amount)
    return _suppress_neighbor_bleed(keys, raw_on, strength), strength


def _key_activation(key: Key, sat: float, val: float) -> tuple[bool, float]:
    sat_strength = sat - key.baseline_sat
    if key.is_black:
        colored = sat >= max(SAT_ON_BLACK, key.baseline_sat + SAT_DELTA_BLACK)
        bright = val >= max(VAL_ON_BLACK, key.baseline_val + VAL_DELTA_BLACK)
        return colored and bright, sat_strength
    colored = sat >= max(SAT_ON, key.baseline_sat + SAT_DELTA)
    dimmed = val <= key.baseline_val - VAL_DELTA_WHITE
    return colored or dimmed, sat_strength


def _is_neighbor_pitch(a: int, b: int) -> bool:
    return abs(a - b) in (1, 2)


def _suppress_neighbor_bleed(
    keys: list[Key], raw_on: list[bool], strength: list[float]
) -> list[bool]:
    """Keep the strongest key among a cluster of neighboring hits."""
    by_midi = {key.midi: i for i, key in enumerate(keys)}
    filtered = list(raw_on)
    for i, key in enumerate(keys):
        if not raw_on[i]:
            continue
        for delta in (-2, -1, 1, 2):
            j = by_midi.get(key.midi + delta)
            if j is None or not raw_on[j] or not _is_neighbor_pitch(key.midi, keys[j].midi):
                continue
            if strength[i] < strength[j] * BLEED_RATIO:
                filtered[i] = False
                break
            if key.is_black and not keys[j].is_black and strength[i] <= strength[j]:
                filtered[i] = False
                break
    return filtered


def _same_envelope(a: Note, b: Note) -> bool:
    duration = min(a.end - a.start, b.end - b.start)
    if duration <= 0:
        return False
    overlap = min(a.end, b.end) - max(a.start, b.start)
    if overlap <= 0:
        return False
    same_span = abs(a.start - b.start) <= 0.08 and abs(a.end - b.end) <= 0.08
    return same_span or overlap >= 0.85 * duration


def _drop_chromatic_ghosts(notes: list[Note]) -> list[Note]:
    """If two neighbors share timing, keep only the stronger one."""
    if len(notes) < 2:
        return notes
    drop: set[int] = set()
    for i, note in enumerate(notes):
        for j, other in enumerate(notes):
            if i == j or not _is_neighbor_pitch(note.pitch, other.pitch):
                continue
            if not _same_envelope(note, other):
                continue
            if note.strength < other.strength:
                drop.add(i)
                break
            if note.strength == other.strength and i > j:
                drop.add(i)
                break
    return [note for i, note in enumerate(notes) if i not in drop]


def _scan_notes(
    cap: cv2.VideoCapture,
    keys: list[Key],
    fps: float,
    frame_count: int,
    progress_cb: ProgressFn | None,
) -> list[Note]:
    n = len(keys)
    state = [False] * n
    pending_on = [0] * n
    pending_off = [0] * n
    started = [None] * n
    strength_acc = [0.0] * n
    strength_n = [0] * n
    notes: list[Note] = []

    index = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        time_s = index / fps
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        raw_on: list[bool] = []
        strength: list[float] = []
        for key in keys:
            sat, val = _sample_key(hsv, key)
            on, amount = _key_activation(key, sat, val)
            raw_on.append(on)
            strength.append(amount)
        raw_on = _suppress_neighbor_bleed(keys, raw_on, strength)

        for i, key in enumerate(keys):
            if raw_on[i]:
                pending_on[i] += 1
                pending_off[i] = 0
                if not state[i] and pending_on[i] >= DEBOUNCE_FRAMES:
                    state[i] = True
                    started[i] = time_s
                    strength_acc[i] = 0.0
                    strength_n[i] = 0
                if state[i]:
                    strength_acc[i] += strength[i]
                    strength_n[i] += 1
            else:
                pending_off[i] += 1
                pending_on[i] = 0
                if state[i] and pending_off[i] >= DEBOUNCE_FRAMES:
                    state[i] = False
                    start = started[i]
                    started[i] = None
                    if start is not None:
                        avg = strength_acc[i] / max(strength_n[i], 1)
                        _maybe_add_note(notes, key.midi, start, time_s, avg)

        index += 1
        if progress_cb and (index % 8 == 0 or (frame_count and index >= frame_count)):
            scanned = index / frame_count if frame_count else min(0.99, index / 5000)
            _report(progress_cb, 0.16 + 0.84 * min(1.0, scanned))

    end_time = index / fps
    for i, key in enumerate(keys):
        if state[i] and started[i] is not None:
            avg = strength_acc[i] / max(strength_n[i], 1)
            _maybe_add_note(notes, key.midi, started[i], end_time, avg)

    notes.sort(key=lambda note: (note.start, note.pitch))
    notes = _drop_chromatic_ghosts(notes)
    _report(progress_cb, 1.0)
    return notes


def _maybe_add_note(
    notes: list[Note], pitch: int, start: float, end: float, strength: float = 0.0
) -> None:
    if end - start >= MIN_NOTE_SEC:
        notes.append(Note(pitch=pitch, start=start, end=end, strength=strength))
