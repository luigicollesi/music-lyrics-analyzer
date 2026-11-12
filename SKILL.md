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

## Markdown lyrics and song structure

Prefer a UTF-8 `.md` lyrics document with an optional `# Song title`, author annotations before the song, and labeled musical sections such as `## Chorus (abertura)`, `## Verse 1`, `## Bridge`, and `## Final Chorus`. Labels and introductory metadata are never sung.

Example:

```bash
python analise.py --music-path "audio/Inferno.wav" --lyrics-path "lyrics/Inferno.md" --language pt --model base --device cpu --lrc
```

Read `sections[]` in the output JSON first. Each section has a distinct `id`, `type`, `label`, `detail`, `line_ids`, `word_ids`, estimated `start`/`end` (or null), and aligned/total word counts. Use `section_id` on lines and words to recover verse/chorus context without guessing based on the wording. The CLI prints musical sections and their aligned time ranges for inspection. Do not treat section headers or author notes as vocals. Timing still requires valid audio alignment.

## Hierarchical alignment (V0.2)

By default, `python analise.py` executes a hierarchical workflow: transcribe overlapping audio windows to find distinctive section anchors, choose monotonic occurrences of repeated choruses, and force-align shorter section/phrase windows with stable-ts. The lyrics Markdown remains authoritative; ASR is only an auxiliary locator. Never infer exact times from BPM or line position.

```bash
python analise.py --music-path "song.wav" --lyrics-path "lyrics.md" --language pt --model base --device cpu --strategy hierarchical --resume --report --lrc
```

Use `--strategy legacy` for the previous global alignment baseline; `--asr-segments` for supplied recognizer segments; `--section-timings` for manually verified section windows; and `--vocals-path` for a pre-separated vocal file preserving the original timeline. `--max-local-seconds` constrains local alignment to avoid long drifting ranges. `--resume` caches the ASR output. None of these flags guarantee correct singing alignment.

Read `diagnostics.sections[]` (coverage, anchor confidence, status), `diagnostics.errors[]` and `sections[].coverage` before deciding when to display lyrics in the video. A section may remain unresolved; do not fill its time by interpolating neighboring sections. Outputs require listening verification before frame-accurate production.
## Read the result

Use `rhythm.bpm`, `rhythm.beats`, `words[].start`, `words[].end`, `words[].strength`, `words[].alignment.status`, and `warnings`. Each word has a unique ID, including repetitions. `strength` is a *track-relative acoustic proxy*, not emotion or objective vocal force. When `analysis.strength_source` is `mixture`, instrument levels can bias it.

Never fabricate missing word timings or confidence. Treat low-confidence/unresolved words as manual-review tasks. Compare the output to actual playback before applying frame-accurate typography, and never time vocals solely from BPM. Do not edit the user's audio.

## Output contract

`analysis/lyrics-analysis.json` is the canonical machine-readable output; optional `.lrc` is only an aid for subtitles. For music-video work, pass JSON to `music-video-director` as timing evidence, not as a substitute for visual review.