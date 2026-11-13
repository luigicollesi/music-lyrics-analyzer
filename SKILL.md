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
## Vocal prosody per word (V0.3; opt-in)

After verified lyric alignment, run `--prosody pyin` to estimate vocal F0, melody contour, RMS and dynamics for each word. This is acoustic measurement, **not** reliable emotion, shouting, breathiness or vibrato classification: unknown delivery properties are always null.

```bash
python analise.py --music-path song.wav --lyrics-path lyrics.md --strategy hierarchical --prosody pyin --pitch-contour --vocals-path vocals.wav --report --resume
```

For deterministic timing tests, supply trusted `--word-timings trusted.json` instead of ASR. The original audio and separated stem must start at exactly the same time and have matching duration. Prefer the vocal stem: mixtures can cause the estimated F0 to track instruments, so output is marked reliability `low` if no stem is provided.

Per-word data is in `words[].vocal` (null if the word is not aligned). Pitch contains median F0 (`mean_hz`), 10th/90th percentile bounds (`min_hz` and `max_hz`), voiced_fraction, voicing_probability, contour (`rising`, `falling`, `steady`, `variable` or null), change in semitones, and optional timestamped F0 samples. Intensity provides RMS/peak in dBFS, attack_db, relative change_db and an energy contour. Signal normalization affects absolute values. Per-frame samples are emitted only with `--pitch-contour` to keep the JSON small.

Pitch inference is bounded to short contiguous groups of already-aligned words. Set `--pitch-min-hz`, `--pitch-max-hz` and `--pitch-hop-length` to accommodate singer pitch range and precision. Missing/unvoiced pitch stays null. Pitch tracking can make octave and voicing errors; verify against isolated vocals and listening before frame-accurate graphics. No timestamps or emotions may be invented. `schema_version` becomes 2 if enabled, otherwise remains 1 for compatibility.

## Read the result

Use `rhythm.bpm`, `rhythm.beats`, `words[].start`, `words[].end`, `words[].strength`, `words[].alignment.status`, and `warnings`. Each word has a unique ID, including repetitions. `strength` is a *track-relative acoustic proxy*, not emotion or objective vocal force. When `analysis.strength_source` is `mixture`, instrument levels can bias it.

Never fabricate missing word timings or confidence. Treat low-confidence/unresolved words as manual-review tasks. Compare the output to actual playback before applying frame-accurate typography, and never time vocals solely from BPM. Do not edit the user's audio.

## Output contract

`analysis/lyrics-analysis.json` is the canonical machine-readable output; optional `.lrc` is only an aid for subtitles. For music-video work, pass JSON to `music-video-director` as timing evidence, not as a substitute for visual review.
## Contextual word recovery (V0.4 candidate)

In hierarchical mode, `--recovery auto` performs a second word-timestamp ASR pass only on bounded local groups with unresolved lyric occurrences. The reference lyrics determine the actual text. Exact/fuzzy transcript anchors are linked monotonically, then a limited number of one-to-one substitutions between reliable anchors may be recovered. A single trailing or leading substitution can be inferred only with two adjacent anchors and some textual similarity.

The output distinguishes `word-asr-exact`, `word-asr-fuzzy`, `word-asr-contextual`, and `word-asr-contextual-edge` in `words[].alignment.method`. Contextual matches have `alignment.status=tentative` and confidence null; do not treat them as acoustic proof of speech or frame-accurate timestamps. Missing ASR words stay unresolved. `--recovery none` switches this pass off. Check `diagnostics.attempts[].word_asr_recovery` and validate results against real audio before using them for VFX.


## V0.5 experimental CTC alignment

Use `--aligner hybrid --strategy hierarchical` to run the regular stable-ts alignment and attempt CTC recovery only on the remaining missing words. Install optional dependencies with `python -m pip install -e '.[align,ctc]'`; this downloads an MMS model on first use. The Portuguese language adapter is `--ctc-language por`. Keep `--aligner stable-ts` as default until accuracy benchmarks are completed. Use `--aligner ctc --section-timings sections.json` only when **every supplied section** is independently verified, at most 20 seconds long, and correctly localized.

The CTC backend produces grapheme-token timing hypotheses, not verified phoneme boundaries; it cannot prove that the lyric was sung. Every `ctc-mms` word is tagged `alignment.status=tentative`; do not treat its model score as calibrated confidence. The model has high memory requirements and its default checkpoint has a noncommercial license. Optional source separation must preserve the original timeline.

After each run compare outputs using `python benchmark_compare.py baseline.json candidate.json`; review newly timed words, lost timings and error in a human-annotated reference sample before enabling frame-exact animation. No automatic merge to main.
