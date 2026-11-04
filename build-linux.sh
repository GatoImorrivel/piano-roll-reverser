#!/usr/bin/env bash
# Bundle the Tkinter app into dist/piano-roll/piano-roll
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -d .venv ]]; then
  echo "Create the project venv first: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install -q -r requirements-build.txt

if [[ -x vendor/ffmpeg ]]; then
  echo "Bundling vendor/ffmpeg into the app."
else
  echo "No vendor/ffmpeg — the built app will use ffmpeg from PATH."
  echo "For a self-contained build, drop static ffmpeg/ffprobe into vendor/ first."
fi

python -m PyInstaller --noconfirm piano-roll.spec

echo
echo "Built: $PWD/dist/piano-roll/piano-roll"
echo "Run:   ./dist/piano-roll/piano-roll"
echo "The target machine still needs PulseAudio/PipeWire (paplay) or ffplay for sound."
echo "If you did not vendor a static ffmpeg, install ffmpeg there too."
