"""YouTube download (H.264 + audio) and local video path checks."""

from __future__ import annotations

import os
import re
import shutil
import webbrowser
from pathlib import Path
from typing import Callable

import yt_dlp

VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}

_URL_RE = re.compile(r"^https?://", re.IGNORECASE)

ProgressFn = Callable[[float], None]

_LOGIN_HINTS = (
    "sign in",
    "login required",
    "confirm you're not a bot",
    "confirm you are not a bot",
    "confirm you’re not a bot",
    "age-restricted",
    "age restricted",
    "members-only",
    "members only",
    "only available to",
    "private video",
    "cookies-from-browser",
    "use --cookies",
)

_COOKIE_DOMAINS = (
    "youtube.com",
    "youtu.be",
    "youtube-nocookie.com",
    "googlevideo.com",
    "google.com",
    "gstatic.com",
)

_BROWSER_PATHS = {
    "firefox": (
        "~/.mozilla/firefox",
        "~/snap/firefox/common/.mozilla/firefox",
        "~/.var/app/org.mozilla.firefox/.mozilla/firefox",
    ),
    "chrome": (
        "~/.config/google-chrome",
        "~/snap/google-chrome/current/.config/google-chrome",
        "~/.var/app/com.google.Chrome/config/google-chrome",
    ),
    "chromium": (
        "~/.config/chromium",
        "~/snap/chromium/common/chromium",
        "~/.var/app/org.chromium.Chromium/config/chromium",
    ),
    "brave": (
        "~/.config/BraveSoftware/Brave-Browser",
        "~/.var/app/com.brave.Browser/config/BraveSoftware/Brave-Browser",
    ),
    "edge": (
        "~/.config/microsoft-edge",
        "~/.var/app/com.microsoft.Edge/config/microsoft-edge",
    ),
    "vivaldi": ("~/.config/vivaldi",),
    "opera": ("~/.config/opera",),
    "safari": ("~/Library/Cookies",),
    "whale": ("~/.config/naver-whale",),
}


class DownloadError(Exception):
    """Raised when a YouTube download or local path check fails."""


class AuthRequired(DownloadError):
    """YouTube asked for a signed-in session (cookies)."""


def is_url(value: str) -> bool:
    return bool(_URL_RE.match(value.strip()))


def config_dir() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    root = Path(xdg) if xdg else Path.home() / ".config"
    return root / "piano-roll-reverse"


def cookie_path() -> Path:
    return config_dir() / "youtube.cookies.txt"


def has_saved_cookies() -> bool:
    path = cookie_path()
    return path.is_file() and path.stat().st_size > 0


def clear_saved_cookies() -> None:
    path = cookie_path()
    if path.is_file():
        path.unlink()


def listed_browsers() -> list[tuple[str, bool]]:
    """Return (browser, looks_installed) for browsers yt-dlp can read."""
    rows: list[tuple[str, bool]] = []
    for name in ("firefox", "chrome", "chromium", "brave", "edge", "vivaldi", "opera"):
        rows.append((name, _browser_looks_installed(name)))
    return rows


def import_browser_cookies(browser: str) -> Path:
    from yt_dlp.cookies import YoutubeDLCookieJar, extract_cookies_from_browser

    try:
        jar = extract_cookies_from_browser(browser)
    except Exception as exc:
        raise DownloadError(
            f"Could not read {browser} cookies: {exc}. "
            "Close the browser and try again, or choose a cookies.txt file."
        ) from exc
    return _save_youtube_cookies(jar)


def import_cookie_file(src: str | Path) -> Path:
    from yt_dlp.cookies import YoutubeDLCookieJar

    src_path = Path(src).expanduser().resolve()
    if not src_path.is_file():
        raise DownloadError(f"Cookie file not found: {src_path}")
    dest = cookie_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src_path != dest.resolve():
        shutil.copy2(src_path, dest)
    try:
        jar = YoutubeDLCookieJar(str(dest))
        jar.load(ignore_discard=True, ignore_expires=True)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise DownloadError(f"Could not read that cookie file: {exc}") from exc
    if not _jar_has_youtube(jar):
        dest.unlink(missing_ok=True)
        raise DownloadError("That file has no YouTube cookies.")
    return dest


def looks_like_login(message: str) -> bool:
    low = message.lower()
    return any(hint in low for hint in _LOGIN_HINTS)


def _browser_looks_installed(name: str) -> bool:
    for raw in _BROWSER_PATHS.get(name, ()):
        if Path(raw).expanduser().exists():
            return True
    return False


def _cookie_domain_ok(domain: str) -> bool:
    host = (domain or "").lstrip(".").lower()
    return any(host == item or host.endswith("." + item) for item in _COOKIE_DOMAINS)


def _jar_has_youtube(jar) -> bool:
    return any(_cookie_domain_ok(getattr(cookie, "domain", "")) for cookie in jar)


def _save_youtube_cookies(jar) -> Path:
    from yt_dlp.cookies import YoutubeDLCookieJar

    kept = YoutubeDLCookieJar()
    for cookie in jar:
        if _cookie_domain_ok(getattr(cookie, "domain", "")):
            kept.set_cookie(cookie)
    if not any(True for _ in kept):
        raise DownloadError(
            "No YouTube cookies found. Open YouTube in that browser, sign in, then import again."
        )
    dest = cookie_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    kept.filename = str(dest)
    kept.save(ignore_discard=True, ignore_expires=True)
    return dest


def _apply_cookies(opts: dict) -> dict:
    path = cookie_path()
    if path.is_file():
        opts["cookiefile"] = str(path)
    return opts


def _prepare_ydl_opts(opts: dict) -> dict:
    from runtime import find_executable

    opts = _apply_cookies(opts)
    ffmpeg = find_executable("ffmpeg")
    if ffmpeg:
        opts["ffmpeg_location"] = str(Path(ffmpeg).parent)
    return opts


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
    opts = _prepare_ydl_opts({
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
    })

    try:
        info = _extract(url, opts)
    except AuthRequired:
        raise
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
        message = _clean_ytdlp_message(exc)
        if looks_like_login(str(exc)) or looks_like_login(message):
            raise AuthRequired(message) from exc
        raise DownloadError(message) from exc
    except AuthRequired:
        raise
    except Exception as exc:  # yt-dlp can raise several types
        message = f"Download failed: {exc}"
        if looks_like_login(str(exc)):
            raise AuthRequired(message) from exc
        raise DownloadError(message) from exc
    if not info:
        raise DownloadError("Could not read video information.")
    return info


def _download_audio_sidecar(url: str, dest: Path, stem: str) -> None:
    opts = _prepare_ydl_opts({
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "outtmpl": str(dest / f"{stem}.audio.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "restrictfilenames": True,
        "overwrites": True,
    })
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
        return "That YouTube video is private. Sign in with an account that can watch it."
    if "members" in text.lower():
        return "This video is members-only. Sign in with an account that has access."
    if "age" in text.lower() and "restrict" in text.lower():
        return "This video is age-restricted. Sign in with an account that can watch it."
    if looks_like_login(text):
        return (
            "YouTube asked you to sign in. "
            "Import cookies from a browser that is logged into YouTube."
        )
    if "Requested format is not available" in text:
        return "YouTube did not offer a downloadable video format. Try a local file."
    return text


def ask_youtube_login(parent=None, *, failed: bool = False) -> bool:
    """Modal: import YouTube cookies from a browser or a cookies.txt file."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    created_root = False
    if parent is None:
        parent = tk.Tk()
        parent.withdraw()
        created_root = True

    result = {"ok": False}

    win = tk.Toplevel(parent)
    win.title("YouTube sign-in")
    win.resizable(True, False)
    win.minsize(460, 280)
    win.transient(parent)
    win.grab_set()

    status_var = tk.StringVar()
    browser_var = tk.StringVar()
    hint = (
        "YouTube asked yt-dlp for a signed-in session. "
        "Password login is not supported — import cookies from a browser "
        "that is already logged into YouTube."
    )
    if failed and has_saved_cookies():
        hint = (
            "The saved YouTube sign-in did not work. "
            "Import cookies again from a browser that can play the video."
        )

    def refresh_status() -> None:
        if has_saved_cookies():
            status_var.set("Signed in (cookies saved on this computer).")
        else:
            status_var.set("Not signed in.")

    def finish(ok: bool) -> None:
        result["ok"] = ok
        win.grab_release()
        win.destroy()

    def do_import_browser() -> None:
        browser = browser_var.get().strip()
        if not browser:
            messagebox.showinfo("Pick a browser", "Choose a browser first.", parent=win)
            return
        try:
            import_browser_cookies(browser)
        except DownloadError as exc:
            messagebox.showerror("Import failed", str(exc), parent=win)
            refresh_status()
            return
        result["ok"] = True
        finish(True)

    def do_import_file() -> None:
        path = filedialog.askopenfilename(
            parent=win,
            title="Choose a Netscape cookies.txt file",
            filetypes=[
                ("Cookie files", "*.txt *.cookies"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        try:
            import_cookie_file(path)
        except DownloadError as exc:
            messagebox.showerror("Import failed", str(exc), parent=win)
            refresh_status()
            return
        result["ok"] = True
        finish(True)

    def do_open_youtube() -> None:
        webbrowser.open("https://www.youtube.com")

    def do_sign_out() -> None:
        clear_saved_cookies()
        refresh_status()

    frame = ttk.Frame(win, padding=16)
    frame.pack(fill=tk.BOTH, expand=True)
    ttk.Label(frame, text="YouTube sign-in", font=("TkDefaultFont", 12, "bold")).pack(
        anchor=tk.W
    )
    ttk.Label(frame, text=hint, wraplength=420).pack(anchor=tk.W, pady=(6, 10))
    ttk.Label(frame, textvariable=status_var).pack(anchor=tk.W, pady=(0, 10))

    ttk.Label(frame, text="Browser already logged into YouTube").pack(anchor=tk.W)
    browsers = listed_browsers()
    first_found = next((name for name, found in browsers if found), browsers[0][0])
    browser_var.set(first_found)
    for name, found in browsers:
        label = f"{name}  (found)" if found else name
        ttk.Radiobutton(frame, text=label, value=name, variable=browser_var).pack(
            anchor=tk.W
        )

    actions = ttk.Frame(frame)
    actions.pack(fill=tk.X, pady=(12, 0))
    ttk.Button(actions, text="Open YouTube…", command=do_open_youtube).pack(side=tk.LEFT)
    ttk.Button(actions, text="Import from browser", command=do_import_browser).pack(
        side=tk.LEFT, padx=(8, 0)
    )
    ttk.Button(actions, text="Choose cookies.txt…", command=do_import_file).pack(
        side=tk.LEFT, padx=(8, 0)
    )

    footer = ttk.Frame(frame)
    footer.pack(fill=tk.X, pady=(12, 0))
    ttk.Button(footer, text="Sign out", command=do_sign_out).pack(side=tk.LEFT)
    ttk.Button(footer, text="Cancel", command=lambda: finish(False)).pack(side=tk.RIGHT)
    ttk.Button(
        footer, text="Continue", command=lambda: finish(has_saved_cookies())
    ).pack(side=tk.RIGHT, padx=(0, 8))

    refresh_status()
    win.protocol("WM_DELETE_WINDOW", lambda: finish(False))
    win.wait_window()
    if created_root:
        parent.destroy()
    return result["ok"]
