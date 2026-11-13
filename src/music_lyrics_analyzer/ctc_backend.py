"""Optional multilingual CTC alignment backend for known lyrics.

No inference modules are imported until explicitly requested. The supplied
lyrics are authoritative; a CTC path is a hypothesis, not proof of singing.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
import unicodedata
import numpy as np


@dataclass(frozen=True)
class Candidate:
    word_id: str
    start: float
    end: float
    score: float
    method: str = "ctc-mms"


def normalize_ctc(text: str) -> str:
    """Normalize Portuguese spelling without modifying the original lyric."""
    value = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in value if
                   (unicodedata.category(c) != "Mn" and
                    (c.isalpha() or c.isdigit())))


def segment_emissions(log_probs: np.ndarray, words: list[dict],
                      tokenizer, *, duration: float, offset: float = 0.,
                      threshold: float = -2.5) -> list[Candidate]:
    """CTC segmentation over *known* word tokens, with strict validation.

    Uses token IDs instead of prepare_text to avoid silently discarding
    unsupported graphemes. Keeps original word IDs and absolute timestamps.
    """
    from ctc_segmentation import (
        CtcSegmentationParameters, prepare_token_list,
        ctc_segmentation, determine_utterance_segments)
    if log_probs.ndim != 2 or not len(log_probs) or duration <= 0:
        return []
    if len(words) == 0:
        return []
    blank_id = tokenizer.pad_token_id
    if blank_id is None or not 0 <= blank_id < log_probs.shape[1]:
        raise ValueError("CTC tokenizer blank/pad ID incompatible with emissions")
    unknown_id = tokenizer.unk_token_id
    tokens = []
    for word in words:
        target = normalize_ctc(word["text"])
        if not target:
            return []
        # Tokenize each word separately; no blank/unknown token is permitted.
        ids = tokenizer(target, add_special_tokens=False).input_ids
        if not ids or any(i == unknown_id or i == blank_id or
                          i < 0 or i >= log_probs.shape[1] for i in ids):
            return []
        tokens.append(np.asarray(ids, dtype=np.int64))
    config = CtcSegmentationParameters()
    config.blank = int(blank_id)
    config.index_duration = float(duration) / len(log_probs)
    config.score_min_mean_over_L = 2
    # CTC segmentation expects log-posterior probabilities.
    matrix, starts = prepare_token_list(config, tokens)
    timing, char_probs, _ = ctc_segmentation(config, log_probs, matrix)
    spans = determine_utterance_segments(
        config, starts, char_probs, timing, [w["text"] for w in words])
    if len(spans) != len(words):
        return []
    result = []
    previous = offset
    for word, (start, end, score) in zip(words, spans):
        a = float(start) + offset
        b = float(end) + offset
        confidence = float(score)
        if not all(map(math.isfinite, (a, b, confidence))):
            continue
        if confidence < threshold or a < offset - .04 or (
            b > offset + duration + .04) or a >= b or b-a > 5.0:
            continue
        if a < previous - .04:
            continue
        result.append(Candidate(word["id"], round(max(a, offset), 3),
                                round(min(b, offset+duration), 3),
                                round(confidence, 4)))
        previous = b
    return result


class MMSBackend:
    """Lazy, process-local MMS model; one model load for many short windows."""

    def __init__(self, model_id="facebook/mms-1b-all",
                 language="por", device="cpu"):
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC
        self.torch = torch
        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            model_id, target_lang=language)
        self.model = Wav2Vec2ForCTC.from_pretrained(
            model_id, target_lang=language,
            ignore_mismatched_sizes=True).to(device).eval()
        # Match tokenizer vocabulary to the loaded language adapter.
        self.processor.tokenizer.set_target_lang(language)
        if hasattr(self.model, "load_adapter"):
            self.model.load_adapter(language)

    def infer(self, audio16: np.ndarray) -> np.ndarray:
        torch = self.torch
        payload = self.processor(
            np.asarray(audio16, dtype=np.float32), sampling_rate=16000,
            return_tensors="pt")
        payload = {k: v.to(self.device) for k, v in payload.items()}
        with torch.inference_mode():
            logits = self.model(**payload).logits[0]
            return torch.log_softmax(logits, dim=-1).cpu().numpy()

    def align(self, audio16, words, start, end, *, threshold=-2.5):
        if end - start > 20.0:
            raise ValueError("CTC window larger than 20 seconds")
        samples = audio16[max(0, int(start*16000)):int(end*16000)]
        if len(samples) < 3200:
            return []
        emission = self.infer(samples)
        return segment_emissions(emission, words, self.processor.tokenizer,
                                 duration=len(samples)/16000.,
                                 offset=start, threshold=threshold)


def recover_ctc_gaps(backend, audio16, words: list[dict],
                     *, start: float, end: float,
                     max_gap_seconds: float = 20.,
                     max_gap_words: int = 40,
                     threshold: float = -2.5) -> dict:
    """CTC only on missing spans bounded by other aligned words or section.

    Never overwrites previous timings. An unanchored lengthy section is left
    unresolved for independent section localization.
    """
    count = 0
    attempts = []
    i = 0
    while i < len(words):
        if words[i]["start"] is not None:
            i += 1
            continue
        j = i
        while j < len(words) and words[j]["start"] is None:
            j += 1
        left = words[i - 1]["end"] if i else start
        right = words[j]["start"] if j < len(words) else end
        left = float(left)
        right = float(right)
        attempt = {"first_word_id": words[i]["id"],
                   "count": j-i, "window": [round(left, 3), round(right, 3)]}
        if not (0 <= left < right <= end+.1) or (
                right-left > max_gap_seconds) or j-i > max_gap_words:
            attempt["status"] = "skipped_window"
            attempts.append(attempt)
            i = j
            continue
        try:
            candidates = backend.align(audio16, words[i:j], left, right,
                                       threshold=threshold)
            by_id = {candidate.word_id: candidate for candidate in candidates}
            for word in words[i:j]:
                candidate = by_id.get(word["id"])
                if candidate is None:
                    continue
                word.update(start=candidate.start, end=candidate.end,
                            duration=round(candidate.end-candidate.start, 3))
                word["alignment"] = {
                    "status": "tentative", "confidence": None,
                    "method": candidate.method,
                    "acoustic_score": candidate.score}
                count += 1
            attempt["recovered"] = len(candidates)
            attempt["status"] = "processed"
        except Exception as exc:
            attempt["status"] = "error"
            attempt["message"] = str(exc)[:250]
        attempts.append(attempt)
        i = j
    return {"recovered": count, "attempts": attempts}
