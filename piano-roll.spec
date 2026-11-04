# -*- mode: python ; coding: utf-8 -*-
"""Linux build: pyinstaller --noconfirm piano-roll.spec"""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = [
    "audio_player",
    "debug_view",
    "detector",
    "downloader",
    "midi_synth",
    "midi_writer",
    "piano_roll_view",
    "runtime",
    "PIL._tkinter_finder",
    "tkinter",
    "tkinter.filedialog",
    "tkinter.messagebox",
    "tkinter.ttk",
]

for package in ("yt_dlp", "pretty_midi", "certifi"):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

root = Path(SPECPATH)
for name in ("ffmpeg", "ffprobe", "ffplay"):
    exe = root / "vendor" / name
    if exe.is_file():
        binaries.append((str(exe), "."))

a = Analysis(
    ["app.py"],
    pathex=[str(root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib", "scipy", "pandas", "IPython", "jupyter"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="piano-roll",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="piano-roll",
)
