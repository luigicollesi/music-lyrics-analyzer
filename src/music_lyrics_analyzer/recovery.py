"""Second-pass recovery of lyric words using observed ASR word timestamps."""
from __future__ import annotations

from math import isfinite

from .contextual_alignment import align_transcript_words
from .core import TOKEN, normalize


def recognize_word_timestamps(model, audio16, start, end, language):
    """Transcribe a bounded vocal region with word timestamps, in absolute time."""
    sr = 16000
    if end <= start or end - start > 35:
        return []
    audio = audio16[int(start * sr):int(end * sr)]
    if len(audio) < sr // 2:
        return []
    result = model.transcribe(audio, language=language,
                              condition_on_previous_text=False,
                              verbose=None, word_timestamps=True)
    output = []
    for segment in (getattr(result, "segments", None) or []):
        for word in (getattr(segment, "words", None) or []):
            value = str(getattr(word, "word", "")).strip()
            pieces = TOKEN.findall(value)
            # A multi-token recognition has only one measured interval.
            # Do not assign that same span to several different words.
            if len(pieces) != 1:
                continue
            try:
                a = float(word.start) + start
                b = float(word.end) + start
            except (TypeError, ValueError, AttributeError):
                continue
            if not (isfinite(a) and isfinite(b) and start <= a < b <= end + .1):
                continue
            output.append({"text": pieces[0], "start": round(a, 3),
                           "end": round(b, 3),
                           "confidence": getattr(word, "probability", None)})
    return sorted(output, key=lambda w: (w["start"], w["end"]))


def recover_with_word_asr(model, audio16, words, start, end, language,
                          *, transcript=None):
    """Fill unresolved lyric occurrences, never replacing earlier alignments.

    ASR recognition must supply word timestamps; the canonical lyric text
    always wins over transcription spelling. Inferred substitutions remain
    marked 'tentative' and do not count as acoustically verified.
    """
    if not words or all(w["start"] is not None for w in words):
        return {"candidates": 0, "recovered": 0, "by_status": {}}
    if transcript is None:
        transcript = recognize_word_timestamps(model, audio16, start, end, language)
    transcript = [t for t in transcript
                  if t.get("start") is not None and t.get("end") is not None
                  and start <= t["start"] < t["end"] <= end + .1]
    aligned = align_transcript_words(words, transcript)
    recovered = 0
    kinds = {}
    for i, (word, candidate) in enumerate(zip(words, aligned)):
        if word["start"] is not None or candidate["start"] is None:
            continue
        # Preserve chronological integrity relative to existing alignments,
        # rather than counting every ASR match as an accurate occurrence.
        earlier = next((w for w in reversed(words[:i])
                        if w["end"] is not None), None)
        later = next((w for w in words[i + 1:]
                      if w["start"] is not None), None)
        a, b = candidate["start"], candidate["end"]
        if earlier and a < earlier["end"] - .04:
            continue
        if later and b > later["start"] + .04:
            continue
        # A contextual word is a text-order hypothesis over an ASR interval.
        status = candidate["status"]
        word.update(start=round(a, 3), end=round(b, 3),
                    duration=round(b - a, 3))
        word["alignment"] = {
            "status": "tentative" if status == "contextual" else "aligned",
            "confidence": None,
            "method": "word-asr-" + status,
        }
        kinds[status] = kinds.get(status, 0) + 1
        recovered += 1
    return {"candidates": len(transcript), "recovered": recovered,
            "by_status": kinds}
