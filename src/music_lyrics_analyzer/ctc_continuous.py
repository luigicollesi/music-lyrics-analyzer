"""CTC Viterbi forced alignment of a *continuous* phrase token sequence.

The implementation keeps O(S) floating-point scores and O(T*S) byte-sized
backpointers (S=2*number_of_tokens+1), avoiding a huge float trellis.
Acoustic CTC paths are hypotheses, never evidence the lyrics were sung.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from .phrase_tokens import PhraseTokens, check_ctc_capacity


@dataclass(frozen=True)
class TokenSpan:
    index: int
    token_id: int
    word_id: str | None
    start_frame: int
    end_frame: int  # exclusive
    mean_logprob: float


def align_ctc_path(log_probs: np.ndarray, target_ids, *, blank_id: int):
    """Return a monotonic state path using CTC collapse semantics.

    States alternate blanks and targets. Transitions are stay, advance-one,
    advance-two, with the last prohibited for repeated neighboring tokens.
    All arithmetic uses log-probabilities. Raises on infeasible targets.
    """
    scores = np.asarray(log_probs, dtype=np.float64)
    target = tuple(int(x) for x in target_ids)
    if scores.ndim != 2 or not scores.shape[0] or not scores.shape[1]:
        raise ValueError("CTC emissions must have shape (frames, vocabulary).")
    if not target:
        raise ValueError("CTC target must contain at least one token.")
    if not 0 <= blank_id < scores.shape[1]:
        raise ValueError("Invalid blank ID.")
    if any(i == blank_id or not 0 <= i < scores.shape[1] for i in target):
        raise ValueError("CTC target contains blank or out-of-vocabulary tokens.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Nonfinite CTC log probabilities.")
    capacity = check_ctc_capacity(target, scores.shape[0])
    if capacity["status"] != "feasible":
        raise ValueError(
            f"insufficient_frames: required {capacity['required_frames']}, "
            f"available {capacity['available_frames']}")

    states = 2 * len(target) + 1
    symbols = np.full(states, blank_id, dtype=np.int64)
    symbols[1::2] = target
    # prev holds maximized prefix log likelihood; steps store only backtracks.
    prev = np.full(states, -np.inf, dtype=np.float64)
    prev[0] = scores[0, blank_id]
    prev[1] = scores[0, target[0]]
    back = np.zeros((len(scores), states), dtype=np.uint8)
    allowed_skip = np.zeros(states, dtype=bool)
    for state in range(3, states, 2):
        allowed_skip[state] = symbols[state] != symbols[state - 2]

    for t in range(1, len(scores)):
        stay = prev
        advance = np.empty_like(prev)
        advance[0] = -np.inf
        advance[1:] = prev[:-1]
        skip = np.full_like(prev, -np.inf)
        skip[allowed_skip] = prev[np.flatnonzero(allowed_skip) - 2]
        # When equal, choose stays first to avoid spurious transitions.
        choices = np.stack((stay, advance, skip))
        choice = np.argmax(choices, axis=0)
        back[t] = choice
        prev = choices[choice, np.arange(states)] + scores[t, symbols]
    terminal = states - 2 if prev[-2] >= prev[-1] else states - 1
    if not np.isfinite(prev[terminal]):
        raise ValueError("No legal CTC path could reach the final token.")
    total_score = float(prev[terminal])
    path = np.empty(len(scores), dtype=np.int32)
    state = terminal
    for t in range(len(scores) - 1, -1, -1):
        path[t] = state
        if t:
            state -= int(back[t, state])
    if state != 0 and state != 1:
        raise ValueError("CTC traceback did not reach a valid initial state.")
    return path, total_score


def token_spans_from_path(log_probs, tokens: PhraseTokens, state_path):
    """Return token spans without inferring silence as a sung vowel."""
    path = np.asarray(state_path)
    if len(path) != len(log_probs):
        raise ValueError("Path and emission durations differ.")
    spans = []
    for idx, (token_id, word_id) in enumerate(zip(
            tokens.token_ids, tokens.token_to_word)):
        frames = np.flatnonzero(path == 2 * idx + 1)
        if len(frames) == 0:
            raise ValueError(f"Token {idx} received no emitting frames.")
        start, end = int(frames[0]), int(frames[-1]) + 1
        quality = float(np.mean(log_probs[frames, token_id]))
        spans.append(TokenSpan(idx, token_id, word_id, start, end, quality))
    return spans


def word_candidates_from_spans(spans, *, frame_duration: float, offset: float,
                               threshold=-2.5):
    """Map emitted token spans to lyric occurrence IDs.

    Word delimiters are not assigned to surrounding words. All timestamps
    refer to observed token frames; blank refinement is a later experiment.
    """
    if frame_duration <= 0 or not math.isfinite(frame_duration):
        raise ValueError("Frame duration must be positive and finite.")
    grouped = {}
    for span in spans:
        if span.word_id is not None:
            grouped.setdefault(span.word_id, []).append(span)
    records = []
    for word_id, items in grouped.items():
        a = offset + min(i.start_frame for i in items) * frame_duration
        b = offset + max(i.end_frame for i in items) * frame_duration
        score = sum(i.mean_logprob * (i.end_frame-i.start_frame)
                    for i in items) / sum(i.end_frame-i.start_frame for i in items)
        reason = "low_score" if score < threshold else None
        records.append({
            "word_id":word_id,"start":round(a,4),"end":round(b,4),
            "score":round(score,4),
            "status":"rejected" if reason else "candidate",
            "rejection_reason":reason,
            "method":"ctc-continuous",
            "token_count":len(items)})
    return records


def align_continuous_emissions(log_probs, words, tokenizer, *,
                               duration: float, offset: float = 0.,
                               threshold: float = -2.5, diagnostic=None):
    from .phrase_tokens import tokenize_phrase
    if diagnostic is not None:
        diagnostic.clear()
        diagnostic["status"] = "started"
    tokens = tokenize_phrase(words, tokenizer, vocabulary_size=log_probs.shape[1])
    capacity = check_ctc_capacity(tokens.token_ids,len(log_probs))
    if diagnostic is not None:
        diagnostic.update(token_count=len(tokens.token_ids),
                          word_count=len(words),capacity=capacity,
                          diacritic_fallback=tokens.used_diacritic_fallback)
    if capacity["status"] != "feasible":
        if diagnostic is not None:
            diagnostic.update(status="insufficient_frames",candidate_records=[])
        return []
    path, score = align_ctc_path(
        log_probs, tokens.token_ids, blank_id=tokenizer.pad_token_id)
    spans = token_spans_from_path(log_probs,tokens,path)
    records = word_candidates_from_spans(
        spans,frame_duration=duration / len(log_probs),
        offset=offset,threshold=threshold)
    if diagnostic is not None:
        diagnostic.update(status="ok",total_path_logprob=round(score,4),
                          candidate_records=records,
                          candidates=sum(r["status"]=="candidate" for r in records),
                          rejected={"low_score":sum(
                              r["rejection_reason"]=="low_score" for r in records)})
    return records
