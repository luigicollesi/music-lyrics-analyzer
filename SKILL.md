---
name: music-lyrics-analyzer
description: Analyze local music with authoritative lyrics to estimate each sung word's timing and measurable vocal delivery (F0, melodic movement, dynamics), with uncertainty for AI video synchronization.
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

## V1 foundation — primary goal and integrity gates (experimental)

**User goal:** Given a local music recording and an authoritative lyric file,
produce the actual onset and offset of **each occurrence** of a sung word,
plus measurable aspects of **how** it is sung: F0, melodic trend, voicing,
relative intensity, duration and vocal attack when supported by the audio.
Maintain an explicit distinction between acoustic measurements and
interpretations such as emotion, breathiness, falsetto or shouting.

### Mandatory analysis order

1. Preserve the original audio and lyric occurrence IDs. Parse sections and
   lines, preserving repetitions. Headers and metadata are never sung.
2. Inspect model integrity and identity before relying on MMS outputs:
   checkpoint, target language `por`, language-specific tokenizer indices,
   CTC blank ID, output-head dimensions and adapter loading path.
   Structural compatibility **does not establish** that the trained
   Portuguese head was restored or that the model works on singing.
3. Validate MMS inference separately using a short, known Portuguese spoken
   utterance and a sung phrase; compare decoded tokens/transcript against
   the expected text. The warning about a newly initialized mismatched
   `lm_head` can occur in documented language-adapter workflows and alone
   is **not proof** of an untrained Portuguese model.
4. Localize sections and phrases using the original audio, ASR as
   auxiliary evidence and preserved lyric IDs. Never invent times from BPM.
   A crop must have enough emitted CTC frames for its complete token
   reference; a line with no timed words must not be forced into the
   zero-length gap at its neighboring anchor.
5. Compare word-level/phrase-level CTC hypotheses as **tentative** data.
   The V0.6 `phrase` mode still delegates word boundaries to
   `determine_utterance_segments`; it is **not** a validated continuous
   token-level word aligner. Do not call it one in a report. Avoid
   confusing CTC acoustic scores with calibrated probabilities.
6. Validate temporal order, boundary accuracy and utterance occurrence.
   Unresolved words remain explicitly unresolved; no guessed
   timestamps, extrapolation from beats or silent conflict suppression.
7. Measure F0 and dynamics on sustained vocal context, preferably using
   a verified time-aligned vocals stem. Associate these frame-level
   measurements with aligned words, preserving null for insufficient
   voiced frames or an untrusted timestamp.
8. Export JSON, optional LRC and diagnostics from the same finalized
   timeline. Verify each JSON with a strict parser; compare the LRC
   entry count with timed words and flag unresolved/conflicted regions.
9. Benchmark against hand-labeled word onset/offset and, separately,
   pitch contours/voicing. Coverage alone is not a quality guarantee.

### Commands currently supported

```bash
# Legacy/baseline alignment plus optional experimental CTC gap/phrase recovery
python analise.py --music-path "song.wav" --lyrics-path "lyrics.md" \
  --aligner hybrid --strategy hierarchical --ctc-language por \
  --ctc-window-mode phrase --ctc-score-window 2 --ctc-device cpu \
  --device cpu --resume --report --lrc --output analysis/result.json

# Analyze measured F0/intensity after timing is available.
python analise.py --music-path "song.wav" --lyrics-path "lyrics.md" \
  --aligner stable-ts --strategy hierarchical \
  --prosody pyin --pitch-contour --vocals-path "vocals.wav" \
  --report --output analysis/prosody.json
```

The hybrid phrase mode records a `*-ctc-candidates.json` audit file.
Read its rejection reasons, `resolver.candidate_decisions` and
`model_integrity`; review `diagnostics.timeline_issues` in the main JSON.
The integrity field means **structural audit**, not transcript accuracy.
Run a strict JSON parse on every generated candidate file. Never publish
frame-accurate word VFX on the basis of tentative CTC candidates alone.

### Roadmap — not yet implemented

The future V1 continuous decoder must: tokenize the entire phrase with an
exact token-to-original-word map; align the continuous CTC sequence using
an auditable dynamic program; handle blank frames without treating pauses
as sustained vowels; compare against the legacy decoder on audio with
manual references; and support abstention before publishing word times.
A future vocal module should separate F0/dynamics from speculative emotion
or technique labels, retain voicing probabilities and use distinct confidence
metrics for timing and pitch. Do not describe these capabilities as present
until implemented and benchmarked.

### Model licenses and reproducibility

The source repository uses MIT; `facebook/mms-1b-all` model weights use a
noncommercial license. Record model revision or content digest, dependency
versions and audio hash in production manifests. Running with `--resume`
does not by itself prove that a cache entry is valid for a changed model
revision. Do not modify or upload private audio without authorization.

## V1 PR2 — phrase tokenization and frame-capacity preflight

The optional `phrase_tokens.py` audit tokenizes the **entire normalized
phrase** using the active MMS Portuguese vocabulary, then verifies that the
word-by-word + explicit delimiter reconstruction yields precisely the same
token ID sequence. It preserves each original word occurrence ID and
associates delimiter tokens with `null`, not a neighboring word. If the
tokenizer handles phrase boundaries differently, this audit **fails closed**
instead of fabricating a token-to-word mapping. Diacritics are only removed
as an explicit fallback when the active vocabulary cannot encode the
original spelling.

A separate CTC frame preflight detects references requiring more frames
than a decoder received; `insufficient_frames` is a structural condition,
not evidence of missing singing. `phrase_windows.py` expands search
regions for completely missing lines when sufficient section context
exists; otherwise records `insufficient_section_context`. It must not
invent a timestamp or cross a verified hard section boundary.

The V0.6 phrase backend still publishes **legacy word-utterance**
candidates and not continuous token-level timings. The verified mapping
is only an experimental diagnostic for the following PR. Consult
`diagnostics.attempts[].ctc_phrase.continuous_reference` to review the
token count, optional diacritic fallback and minimum-frame capacity.

### Optional real-model smoke test

After installing `.[align,ctc]`, validate the actual Portuguese decoder
using a human-transcribed **short, isolated** spoken or sung WAV:

```bash
python scripts/validate_mms.py --audio "speech-pt.wav" \
  --expected "frase falada exatamente neste áudio" --device cpu
```

Repeat with a sung phrase. This performs unrestricted CTC decoding and
reports the expected text, predicted text and sample-specific WER.
It does not prove accurate word boundaries or good performance on
full songs. It downloads the checkpoint on first use, and does not
upload local audio. Never run this benchmark on an arbitrary track
with unrelated expected text. A low WER on speech alone does not
establish singing performance.

## V1 PR3 — continuous CTC Viterbi experimental comparison

The `--ctc-window-mode continuous` option now runs a true monotonic CTC
Viterbi path on the **whole phrase token sequence**. Token IDs are mapped
back to occurrence-specific lyric word IDs; delimiter tokens are not
assigned to words. Consecutive identical tokens require a separating CTC
blank. The decoder stores one byte per time/state backpointer and only
two floating-point DP rows, improving memory usage on CPU.

```bash
python analise.py --music-path "song.wav" --lyrics-path "lyrics.md" \
  --aligner hybrid --strategy hierarchical \
  --ctc-window-mode continuous --ctc-device cpu \
  --resume --report --lrc --output analysis/continuous-review.json
```

**The continuous mode is comparison-only.** It records acoustic hypotheses
in `*-ctc-candidates.json` and reports `experimental_only: true` in
`diagnostics.ctc_summary`, but does NOT overwrite the stable-ts baseline.
Scores are mean token-emission log probabilities, **not comparable** to
legacy `ctc-segmentation` scores, nor calibrated confidence. Separate
human validation and a consistent evaluation set are required before any
new timing can become accepted.

The first/last emitted token frames define a candidate interval; long
internal blanks are not evidence that a vowel was sustained. Blank-frame
refinement, acoustic boundary calibration, global soft-anchor selection,
and per-word melodic/vocal metrics improvements remain future work.
Use `--prosody pyin` for the current independent vocal-pitch baseline,
preferably with a synchronized vocal stem.

## V1 PR4a — structural phrase-location conflict audit

In continuous comparison mode, `phrase_conflicts.py` flags an **entirely
unanchored lyric line** when its hypothesized CTC word spans substantially
overlap the measured interval of a neighboring, already anchored lyric
line. This is the reported failure involving "Espinhos rasgam..." and the
following "Três feras...": a valid CTC Viterbi path on the wrong crop is
not evidence of correct lyrics.

The guard is conservative and **diagnostic**, not an acoustic localization
solution. It compares neighboring line occurrence IDs rather than the
strings of repeated choruses. The threshold is an overlap *fraction*,
not a calibrated probability of a correct alignment. An automatic
alignment anchor may also be wrong; avoid promoting its authority without
independent evidence. Candidates that would otherwise pass CTC scoring
but fail this check are marked `rejected` with
`rejection_reason: phrase_location_conflict`; candidates already rejected
on acoustic grounds retain their original reason. Raw timing and scores
remain available. Review `phrase_conflicts` in the CTC candidate JSON
and `diagnostics.ctc_summary.phrase_conflicts`.

**Not yet implemented:** independently finding the correct occurrence of
a fully unanchored phrase using a vocal-activity/ASR search. This guard
cannot guarantee either valid word timings or correct section placement.
The `continuous` mode remains comparison-only, never committing
timestamps automatically. Continue to assess word timing and vocal
intonation as separate goals.

## V1 PR4b — overlapping CTC review (experimental)

Use `--ctc-window-mode overlap --ctc-overlap-seconds 8 --ctc-overlap-hop 4`
to compare candidate word boundaries using overlapping acoustic windows.
The mode selects **only** occurrence-specific lyric lines with at least
two pre-existing word anchors in a crop that contains their entire timed
span away from both crop edges. A line with no trusted localization is
marked `insufficient_independent_anchors`; do **not** force its lyrics
against arbitrary audio or confuse repetition with independent evidence.
The method currently checks crop stability, **not** whether a phrase is
present: independent ASR localization for fully missing lines is future
work.

The decoder uses the existing continuous Viterbi, and a bounded
four-entry in-process emission cache avoids repeat MMS inference
for repeated crops, even without `--resume`. Timing hypotheses are
reported per original occurrence and are **never committed**.
Read `stability` and `phrase_conflicts` in the candidate report.
Multiple agreeing crops from one checkpoint are not independent acoustic
confirmations. Verify improvements with manually marked song timestamps.

```bash
python analise.py --music-path "song.wav" --lyrics-path "lyrics.md" \
  --aligner hybrid --strategy hierarchical --ctc-window-mode overlap \
  --ctc-overlap-seconds 8 --ctc-overlap-hop 4 \
  --ctc-device cpu --resume --report \
  --output analysis/overlap-review.json
```

## V1 missing-line ASR localization in overlap review

The experimental overlap mode now optionally uses the **already generated
or cached** auxiliary ASR segments to propose a crop for a completely
unanchored line. It requires strong ordered lexical overlap and enough
unoccupied audio **between neighboring anchored lyric lines**. It skips
short, ambiguous or repeatedly matched phrases instead of forcing the
reference text onto the wrong singing. All proposals include
`localization_source: asr_lexical_match` and the raw ASR similarity score.

Candidate crop limits and local ordering are safeguards, not a guarantee
that an ASR phrase is correct. Crucially, such proposals are fed to
continuous CTC only for experimental comparison and are **never applied
to final timestamps**. When ASR evidence is absent, the system still
retains the anchored-only overlap behavior. Reusing ASR cache and MMS
emissions avoids adding a new mandatory model or download.

Check `asr_localization` in the output report, examine `skipped`
attempts (especially `ambiguous_asr_occurrence` and
`no_strong_asr_match`), and validate correct location by listening
to the actual crop before changing the accepted timeline.

## Experimental Qwen3-ASR auxiliary backend on NVIDIA GPU

Install optionally: `python -m pip install -e '.[align,ctc,qwen]'`.
Use `--asr-backend qwen --qwen-model 0.6b --qwen-device cuda`
for CUDA or switch to `1.7b` for a larger checkpoint on sufficient
hardware. Qwen processes 16-second overlapping crops with an 8-second
hop by default; each ASR record is **only bounded by its entire crop**,
not word-level timing. It is used to propose locations; all accepted
word timings still require acoustic confirmation.

Example:
```bash
python analise.py --music-path audio/Inferno.wav \
 --lyrics-path lyrics/Inferno.md --aligner hybrid --strategy hierarchical \
 --asr-backend qwen --qwen-model 0.6b --qwen-device cuda \
 --ctc-window-mode overlap --ctc-device cpu --device cpu \
 --resume --report --output analysis/qwen-gpu-review.json
```

**Important legacy GPU caveat:** GTX 1050 Ti is Pascal, compute capability
6.1. Modern CUDA 13 PyTorch wheels do not contain kernels for this GPU.
The loader checks the *installed* CUDA architecture list before loading
Qwen and refuses an unsupported CUDA request rather than silently running
on CPU. A CUDA 12.x PyTorch build with Pascal-compatible kernels is necessary.
An `sm_60` cubin can run on an `sm_61` GPU (same major architecture,
newer device minor revision). The remaining VRAM may still be insufficient. Do not
install FlashAttention 2 or vLLM on Pascal. Qwen's source FP16 support
on old kernels is unverified here; use `--qwen-device cpu` if CUDA errors.
The large model is not expected to fit in 4 GiB VRAM without additional
quantization/offload engineering.

This integration uses the official `qwen-asr` Transformers backend;
it neither installs model weights by default nor changes the baseline
`--asr-backend whisper`. Full pipeline accuracy and actual CUDA
inference still need independent validation on the target machine.

## Correction: coarse Qwen ASR chunks are not timing anchors

The Qwen3-ASR adapter returns transcription text bounded by 16-second
overlapping audio **crops**. These timestamps are not the word/phrase
boundaries required by hierarchical section localization. Earlier code
replaced the short Whisper ASR segment list with those coarse crops, often
yielding **zero anchored sections and 0 timed words** despite successful
Qwen GPU inference.

The Qwen mode now generates and caches Qwen transcripts separately
(`.cache/qwen-*.json`) as supplemental text evidence, while using short
Whisper speech segments to establish and validate the section timing
anchors. It never substitutes the coarse Qwen crop interval for a precise
phrase location. Qwen transcript examples and provenance are exported
in `diagnostics.qwen_asr`; evaluation of those transcripts is currently
diagnostic only. Qwen by itself cannot yet improve published alignment
quality. The additional Whisper ASR pass is intentional to avoid regression.
When `--asr-segments` is explicitly provided, those supplied segments
remain the timing reference.

Do not mistake model loading or repeated `pad_token_id` messages for
proof of a correct ASR transcript. Compare the saved Qwen chunk texts
against the sung lyrics, then develop a window-internal localization
algorithm if the transcript improves recognition.

## Whole-song Qwen transcription experiment (default Qwen mode)

`--asr-backend qwen --qwen-mode full` submits the entire mono 16-kHz song
to a single Qwen3-ASR transcription call, with no externally imposed
16-second windowing. The backend may still have internal audio handling.
The result is saved to `analysis/.cache/qwen-full-*.json` when `--resume`
is supplied, and included in output diagnostics as
`diagnostics.qwen_asr.transcript`.

Example for the target GPU:

```bash
python analise.py --music-path audio/Inferno.wav --lyrics-path lyrics/Inferno.md \
 --aligner hybrid --strategy hierarchical --asr-backend qwen \
 --qwen-mode full --qwen-model 0.6b --qwen-device cuda \
 --qwen-max-new-tokens 2048 --ctc-window-mode overlap \
 --ctc-device cpu --device cpu --resume --report \
 --output analysis/lyrics-analysis-qwen-full.json
```

The generation token limit is a **budget**, not proof of an untruncated
transcription. The `token_limit_risk` field is a rough warning based
on word count, not an authoritative model stopping-reason signal.
Inspect all text, especially the song ending; if truncated, increase
`--qwen-max-new-tokens`. On GTX 1050 Ti, 332 seconds in one request
may exceed available GPU memory. Explicitly retry with
`--qwen-mode windowed` if necessary. There is NO automatic silent
fallback or silent truncation handling.

The full transcript has **no phrase timestamps**. As before, short
Whisper speech segments supply hierarchical timing anchors. The full
Qwen transcript is diagnostic only pending safe full-text-to-lyric
alignment. Hence this feature does not by itself increase coverage.

## Experimental CPU/GPU offloading for full-song Qwen on Pascal

`--qwen-device hybrid` uses Hugging Face Accelerate via
`Qwen3ASRModel.from_pretrained(..., device_map="auto",
max_memory={0:"2560MiB","cpu":"12288MiB"})`. The Qwen wrapper forwards
these arguments to Transformers AutoModel, so this caps model **weight
placement**; it is not an overall CUDA usage limit. The default GPU
budget is 2.5 GiB, RAM budget 12 GiB. Tune with
`--qwen-gpu-memory-gb` and `--qwen-cpu-memory-gb`.

```bash
python analise.py --music-path audio/Inferno.wav --lyrics-path lyrics/Inferno.md \
 --aligner hybrid --strategy hierarchical --asr-backend qwen \
 --qwen-mode full --qwen-model 0.6b --qwen-device hybrid \
 --qwen-gpu-memory-gb 2.5 --qwen-cpu-memory-gb 12 \
 --qwen-max-new-tokens 2048 --ctc-window-mode overlap \
 --ctc-device cpu --device cpu --resume --report \
 --output analysis/lyrics-analysis-qwen-hybrid.json
```

**Experimental limitation:** The official Qwen ASR wrapper assumes one
model device for audio inputs and exposes the underlying model's
`hf_device_map`. Some mixed CPU/GPU placement configurations may fail
during audio encoder or generation because of cross-device tensors.
Most importantly, the previous 1.32 GiB CUDA allocation for a 332-second
input can still occur: weight offloading does not offload intermediate
activations automatically. A 4 GiB Pascal GPU may remain unable to
process the full song in one call. If it OOMs, use
`--qwen-device cpu --qwen-mode full` for the complete transcript or
`--qwen-device cuda --qwen-mode windowed` for CUDA batches.
No automatic fallback; no claims of verified real-device execution.
The cache key distinguishes full mode and GPU/CPU budget settings.

## True parameter streaming: `--qwen-device streamed`

With `--qwen-mode full --qwen-device streamed`, the Qwen3-ASR model is
initially loaded in host RAM, then `accelerate.cpu_offload(model.model,
execution_device="cuda:0")` attaches per-forward hooks that copy module
weights to the GPU when needed and offload them again after each forward.
Unlike static `device_map="auto"`, this streams parameters. The model's
wrapper input device is refreshed to CUDA after hook installation.

```bash
python analise.py --music-path audio/Inferno.wav --lyrics-path lyrics/Inferno.md \
 --aligner hybrid --strategy hierarchical --asr-backend qwen \
 --qwen-mode full --qwen-model 0.6b --qwen-device streamed \
 --qwen-max-new-tokens 2048 --ctc-window-mode overlap \
 --ctc-device cpu --device cpu --resume --report \
 --output analysis/qwen-full-streamed.json
```

This is NOT an activation offload strategy. The 332-second input
previously required an additional ~1.32 GiB CUDA allocation even after
the model was loaded; the same allocation may still fail on a 4 GiB
GPU. Streaming parameters can be slow on PCIe and the Qwen wrapper
may assume unsupported device placement for some internal layers.
Do not claim success without actual hardware validation. If `streamed`
still OOMs, **full CPU inference is the supported fallback preserving
whole-song transcription**; a deeper architecture-specific refactor
would be needed to offload or tile audio-encoder activations.

## Managed runtime foundation — diagnostic milestone

Use `--qwen-memory-report` with `--qwen-mode full` to record checkpoints
before model load, after loading, after offload-hook setup, and after
transcription. CUDA measurements include allocated/reserved/peak/free VRAM;
Linux CPU measurement includes process maximum resident set size. On CUDA
OOM the checkpoints are appended to the raised error for troubleshooting.

The new `inference_runtime/` package separates weight/offload checks,
device validation, profiling and future activation/generation policies.
Importing it does not download a model. It validates that tensors on the
`meta` device have Accelerate hooks attached (directly or on a child).

Example:
```bash
python analise.py --music-path audio/Inferno.wav --lyrics-path lyrics/Inferno.md \
 --aligner hybrid --strategy hierarchical --asr-backend qwen \
 --qwen-mode full --qwen-model 0.6b --qwen-device streamed \
 --qwen-memory-report --ctc-device cpu --device cpu \
 --resume --report --output analysis/qwen-managed.json
```

**Scope:** phase-1 diagnostics and runtime modularization; NO solution
to the previously observed 1.32 GB activation OOM is claimed. The
activation and generation managers currently document unchanged policies.
The Qwen input device proxy is still experimental. Future audio-encoder
changes must prove semantic equivalence before publication. A successful
offline test suite does not validate a 332-second GPU run.

## Phase 2: Audio encoder VRAM profiling

For Qwen's `--qwen-mode full`, add `--qwen-memory-report
--qwen-encoder-memory-trace`. This attaches temporary read-only forward
hooks to the Qwen audio tower and its layers, sampling CUDA allocated,
reserved and peak allocated MiB before/after each module forward.
Hooks are removed even on exceptions, and events are capped at 160.
When CUDA OOM occurs, sampled trace events are appended to the error.
If no known encoder module is discovered, the trace explicitly reports
`audio_encoder_not_found` rather than inventing measurements.
Successful full transcripts contain the trace under
`diagnostics.qwen_asr.transcript.encoder_memory_trace`.

**Limitations:** CPU and GPU operations can be asynchronous; samples are
not a kernel-level memory allocation trace. They may miss a transient
allocation that fails inside a layer; in that case compare the last
pre-forward sample with CUDA OOM allocation size and run a targeted
PyTorch memory snapshot. The profiler does not change Qwen architecture,
activations, attention implementation or language-model KV cache.

Example using 332s audio: add `--qwen-encoder-memory-trace` to the
existing `--qwen-mode full --qwen-device streamed` command.
