"""Conservative word-level lyric/transcript matching for noisy sung ASR.

The lyrics are authoritative for spelling; the ASR provides observed time spans.
Contextual substitutions are never presented as acoustic verification.
"""
from __future__ import annotations

from difflib import SequenceMatcher
from math import isfinite
from .core import normalize


def _score(a: str, b: str) -> float:
    if a == b:
        return 1.0
    # Short function words are dangerous fuzzy anchors ("de" vs "me").
    if min(len(a), len(b)) < 4:
        return 0.0
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def align_transcript_words(lyrics: list[dict], transcript: list[dict],
                           *, fuzzy_threshold: float = 0.78,
                           max_context_gap: int = 4) -> list[dict]:
    """Match in order, then recover *bounded* one-to-one substitutions.

    Each transcript item needs word-level start/end from the recognizer.
    Segment-only timing is rejected; never distribute segment time by guess.
    Results correspond to lyric occurrences and distinguish exact/fuzzy/
    contextual/unresolved. A contextual match is an inference, not verification.
    """
    observed = []
    for original_index, item in enumerate(transcript):
        token = normalize(str(item.get("text", "")))
        try:
            start, end = float(item["start"]), float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not token or not (isfinite(start) and isfinite(end) and 0 <= start < end):
            continue
        observed.append((original_index, token, start, end))
    reference = [normalize(str(w["text"])) for w in lyrics]
    n, m = len(reference), len(observed)
    # Global sequence alignment with skipped tokens. Strong matches are worth
    # keeping; substitutions are recovered only inside two reliable anchors.
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    move = [[None] * (m + 1) for _ in range(n + 1)]
    penalty = 0.37
    for i in range(1, n + 1):
        dp[i][0], move[i][0] = -i * penalty, "lyric"
    for j in range(1, m + 1):
        dp[0][j], move[0][j] = -j * penalty, "asr"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = _score(reference[i - 1], observed[j - 1][1])
            options = [(dp[i - 1][j] - penalty, "lyric"),
                       (dp[i][j - 1] - penalty, "asr")]
            if s >= fuzzy_threshold:
                options.append((dp[i - 1][j - 1] + 2 * s, "match"))
            dp[i][j], move[i][j] = max(options, key=lambda x: x[0])
    anchors = []
    i, j = n, m
    while i or j:
        direction = move[i][j]
        if direction == "match":
            anchors.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif direction == "lyric":
            i -= 1
        else:
            j -= 1
    anchors.reverse()
    result = [{"id": word.get("id"), "text": word["text"],
               "start": None, "end": None, "asr_index": None,
               "status": "unresolved", "similarity": None}
              for word in lyrics]

    def assign(a: int, b: int, status: str) -> None:
        idx, heard, start, end = observed[b]
        result[a].update(start=start, end=end, asr_index=idx, status=status,
                         similarity=round(_score(reference[a], heard), 3))

    for a, b in anchors:
        assign(a, b, "exact" if reference[a] == observed[b][1] else "fuzzy")
    # Only enclosed gaps: an ASR insertion/deletion makes the number of
    # words disagree, so no speculative assignment is made.
    for (left_a, left_b), (right_a, right_b) in zip(anchors, anchors[1:]):
        lyric_gap = right_a - left_a - 1
        asr_gap = right_b - left_b - 1
        if not (0 < lyric_gap == asr_gap <= max_context_gap):
            continue
        for delta in range(1, lyric_gap + 1):
            assign(left_a + delta, left_b + delta, "contextual")
    # One leading/trailing substitution can be recovered when two consecutive
    # anchors establish direction and the unpaired token is still reasonably
    # similar. Never infer a timestamp from an entirely missing ASR token.
    if len(anchors) >= 2:
        first, second = anchors[0], anchors[1]
        if first == (1, 1) and second == (2, 2):
            if _score(reference[0], observed[0][1]) >= .45:
                assign(0, 0, "contextual-edge")
        penultimate, last = anchors[-2], anchors[-1]
        if last == (n - 2, m - 2) and penultimate == (n - 3, m - 3):
            if _score(reference[-1], observed[-1][1]) >= .45:
                assign(n - 1, m - 1, "contextual-edge")
    return result
