#!/usr/bin/env bash
# Make 20 computer-voice clips (flite), 2 per command: LIBERO's wording, and the same with "please" in front.
# Output: data/share_audio/cNN_k.raw, 16 kHz mono 16-bit raw (k = 0 plain, 1 with "please"). No GPU.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONDONTWRITEBYTECODE=1

out=data/share_audio
mkdir -p "$out"

# Read the 10 commands from tasks.py so the wording has one source.
mapfile -t commands < <(python3 -c 'from tasks import COMMANDS; print("\n".join(COMMANDS))')

for i in "${!commands[@]}"; do
  plain="${commands[$i]}"
  for k in 0 1; do
    text="$plain"
    if [ "$k" = 1 ]; then text="please $plain"; fi
    name=$(printf "c%02d_%d" "$i" "$k")
    ffmpeg -nostdin -y -loglevel error -f lavfi -i "flite=text='$text':voice=slt" \
      -ar 16000 -ac 1 -f s16le "$out/$name.raw"
  done
done
echo "made ${#commands[@]} x 2 clips in $out"
