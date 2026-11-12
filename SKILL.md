---
name: music-lyrics-analyzer
description: Analyze local song audio with supplied lyrics to estimate BPM, align each sung word and report relative acoustic intensity for AI video timing.
license: MIT
---

# Music Lyrics Analyzer

Use for requests requiring **song BPM**, word-level lyric timestamps or relative vocal/acoustic emphasis. Require both a local music file and the authoritative lyrics file. Work locally: do not upload private audio to third-party APIs.

## Setup

From the repository root, ensure Python 3.11/3.12 and FFmpeg; install `python -m pip install -e '.[align]'` to activate stable-ts. CPU is default. Model weights may download during the first run. Optional `.[separation]` installs Demucs (not enabled by default).

## Analyze

```bash
python analise.py --music-path "path/to/song.wav" --lyrics-path "path/to/lyrics.txt" --language pt --model base --device cpu --lrc --output analysis/lyrics-analysis.json
```

For testing without inference: use `--aligner none` (word timestamps stay null) or `--word-timings manually-verified.json` (the input timestamps must be real and in chronological order).

## Read the result

Use `rhythm.bpm`, `rhythm.beats`, `words[].start`, `words[].end`, `words[].strength`, `words[].alignment.status`, and `warnings`. Each word has a unique ID, including repetitions. `strength` is a *track-relative acoustic proxy*, not emotion or objective vocal force. When `analysis.strength_source` is `mixture`, instrument levels can bias it.

Never fabricate missing word timings or confidence. Treat low-confidence/unresolved words as manual-review tasks. Compare the output to actual playback before applying frame-accurate typography, and never time vocals solely from BPM. Do not edit the user's audio.

## Output contract

`analysis/lyrics-analysis.json` is the canonical machine-readable output; optional `.lrc` is only an aid for subtitles. For music-video work, pass JSON to `music-video-director` as timing evidence, not as a substitute for visual review.