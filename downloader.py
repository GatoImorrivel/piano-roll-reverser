"""YouTube download (H.264 + audio) and local video path checks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

import yt_dlp

VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)

ProgressFn = Callable[[float], None]


class DownloadError(Exception):
    """Raised when a YouTube download or local path check fails."""


def is_url(value: str) -> bool:
    return bool(_URL_RE.match(value.strip()))


def validate_local_video(path: str) -> Path:
    video = Path(path).expanduser().resolve()
    if not video.is_file():
        raise DownloadError(f"File not found: {video}")
    if video.suffix.lower() not in VIDEO_SUFFIXES:
        raise DownloadError(
            f"Unsupported file type '{video.suffix}'. "
            f"Use {', '.join(sorted(VIDEO_SUFFIXES))}."
        )
    return video


def download_youtube(
    url: str,
    dest_dir: str | Path,
    progress_cb: ProgressFn | None = None,
) -> tuple[Path, str]:
    """Download a single video file. Returns (path, title)."""
    url = url.strip()
    if not is_url(url):
        raise DownloadError("Enter a valid http(s) URL.")

    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    def hook(event: dict) -> None:
        if progress_cb is None or event.get("status") != "downloading":
            return
        total = event.get("total_bytes") or event.get("total_bytes_estimate") or 0
        downloaded = event.get("downloaded_bytes") or 0
        if total:
            progress_cb(min(1.0, downloaded / total))

    # Prefer H.264 + audio remuxed to mp4 so OpenCV can open the picture
    # and ffplay can play the soundtrack. Fall back to a video-only file
    # plus a sidecar audio track if merge is unavailable.
    video_fmt = (
        "bestvideo[height<=1080][vcodec^=avc][protocol^=http]/"
        "bestvideo[height<=1080][vcodec^=avc]/"
        "bestvideo[vcodec^=avc]/"
        "bestvideo[height<=1080][vcodec^=vp9][protocol^=http]/"
        "bestvideo[height<=1080][protocol^=http]/"
        "bestvideo[height<=1080]/"
        "bestvideo/"
        "best"
    )
    merged_fmt = (
        "bestvideo[height<=1080][vcodec^=avc]+bestaudio[ext=m4a]/"
        "bestvideo[height<=1080][vcodec^=avc]+bestaudio/"
        "bestvideo[vcodec^=avc]+bestaudio/"
        "bestvideo[height<=1080]+bestaudio/"
        "bestvideo+bestaudio/"
        "best"
    )
    opts = {
        "format": merged_fmt,
        "outtmpl": str(dest / "%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "restrictfilenames": True,
        "progress_hooks": [hook],
        "overwrites": True,
    }

    try:
        info = _extract(url, opts)
    except DownloadError:
        opts["format"] = video_fmt
        opts.pop("merge_output_format", None)
        info = _extract(url, opts)
        path = _resolve_download_path(info, dest)
        if path is None or not path.is_file():
            raise DownloadError("Download finished but the video file is missing.")
        _download_audio_sidecar(url, dest, path.stem)
        if progress_cb:
            progress_cb(1.0)
        title = (info.get("title") or path.stem).strip() or path.stem
        return path, title

    if not info:
        raise DownloadError("Could not read video information.")

    path = _resolve_download_path(info, dest)
    if path is None or not path.is_file():
        raise DownloadError("Download finished but the video file is missing.")

    if progress_cb:
        progress_cb(1.0)

    title = (info.get("title") or path.stem).strip() or path.stem
    return path, title


def _extract(url: str, opts: dict) -> dict:
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except yt_dlp.utils.DownloadError as exc:
        raise DownloadError(_clean_ytdlp_message(exc)) from exc
    except Exception as exc:  # yt-dlp can raise several types
        raise DownloadError(f"Download failed: {exc}") from exc
    if not info:
        raise DownloadError("Could not read video information.")
    return info


def _download_audio_sidecar(url: str, dest: Path, stem: str) -> None:
    opts = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": str(dest / f"{stem}.audio.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "restrictfilenames": True,
        "overwrites": True,
    }
    try:
        _extract(url, opts)
    except DownloadError:
        return


def _resolve_download_path(info: dict, dest: Path) -> Path | None:
    downloads = info.get("requested_downloads") or []
    if downloads:
        filepath = downloads[0].get("filepath") or downloads[0].get("filename")
        if filepath:
            return Path(filepath)

    video_id = info.get("id")
    ext = info.get("ext")
    if video_id and ext:
        candidate = dest / f"{video_id}.{ext}"
        if candidate.is_file():
            return candidate

    if video_id:
        matches = sorted(dest.glob(f"{video_id}.*"))
        if matches:
            return matches[0]
    return None


def _clean_ytdlp_message(exc: Exception) -> str:
    text = str(exc).strip() or "YouTube download failed."
    text = re.sub(r"^ERROR:\s*", "", text)
    if "Video unavailable" in text:
        return "That YouTube video is unavailable."
    if "Private video" in text:
        return "That YouTube video is private."
    if "Sign in" in text:
        return "YouTube asked for a sign-in; try a local file instead."
    if "Requested format is not available" in text:
        return "YouTube did not offer a downloadable video format. Try a local file."
    return text
