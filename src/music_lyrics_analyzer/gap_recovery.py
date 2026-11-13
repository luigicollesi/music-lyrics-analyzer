"""Acoustic retry for lyric gaps bounded by trusted local evidence.

Never infer word timestamps from the number of characters or BPM.
"""
from __future__ import annotations

from .core import apply_alignment
from .localization import validate_word_timeline


def retry_unresolved_gaps(model, audio16, words, group_start, group_end,
                          language, *, max_gap_seconds=12.0,
                          max_gap_words=18):
    """Re-align only missing consecutive lyric words in a bounded audio crop.

    Uses neighboring already-aligned words to form safe absolute windows.
    The first and last gaps may use group boundaries if the crop is short.
    A successful attempt must preserve chronological ordering. The return
    dictionary is diagnostic: alignment coverage does not imply precision.
    """
    recovered = 0
    attempts = []
    n = len(words)
    i = 0
    while i < n:
        if words[i]["start"] is not None:
            i += 1
            continue
        j = i
        while j < n and words[j]["start"] is None:
            j += 1
        left = words[i - 1] if i else None
        right = words[j] if j < n else None
        # Avoid extrapolating across very large gaps or beyond the song crop.
        begin = float(left["end"]) if left and left.get("end") is not None else float(group_start)
        finish = float(right["start"]) if right and right.get("start") is not None else float(group_end)
        missing = words[i:j]
        attempt = {"first_word": missing[0]["id"], "count": len(missing),
                   "window": [round(begin, 3), round(finish, 3)]}
        if (not (0 <= begin < finish) or
            finish - begin > max_gap_seconds or
            len(missing) > max_gap_words or
            finish - begin < .09):
            attempt["status"] = "skipped_unbounded"
            attempts.append(attempt)
            i = j
            continue
        region = audio16[max(0, int(begin * 16000)):int(finish * 16000)]
        if len(region) < 1600:
            attempt["status"] = "skipped_short_audio"
            attempts.append(attempt)
            i = j
            continue
        try:
            # Input is the authoritative lyric text, not noisy ASR output.
            result = model.align(region, " ".join(w["text"] for w in missing),
                                 language=language, original_split=True)
            proposed = []
            for seg in (getattr(result, "segments", None) or []):
                for item in (getattr(seg, "words", None) or []):
                    a = getattr(item, "start", None)
                    b = getattr(item, "end", None)
                    proposed.append({
                        "text": getattr(item, "word", ""),
                        "start": float(a) + begin if a is not None else None,
                        "end": float(b) + begin if b is not None else None,
                        "confidence": getattr(item, "probability", None),
                    })
            apply_alignment(missing, proposed, method="stable-ts-gap-retry",
                            duration=group_end)
            # Local chronological check and explicit crop bounds.
            for word in missing:
                if word["start"] is None:
                    continue
                if word["start"] < begin - .04 or word["end"] > finish + .04:
                    word.update(start=None, end=None, duration=None,
                                strength=None, features=None)
                    word["alignment"] = {
                        "status": "unresolved", "confidence": None,
                        "method": None,
                    }
            validate_word_timeline(missing)
            count = sum(w["start"] is not None for w in missing)
            attempt.update(status="recovered" if count else "unresolved",
                           recovered=count)
            recovered += count
        except Exception as exc:
            attempt.update(status="error", error=str(exc)[:180])
        attempts.append(attempt)
        i = j
    return {"recovered": recovered, "attempts": attempts}
