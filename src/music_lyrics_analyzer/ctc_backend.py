"""Optional multilingual CTC alignment backend for known lyrics.

No inference modules are imported until explicitly requested. The supplied
lyrics are authoritative; a CTC path is a hypothesis, not proof of singing.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import os
import tempfile
import math
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
                      threshold: float = -2.5, diagnostic: dict | None = None,
                      score_window: int = 2) -> list[Candidate]:
    """CTC segmentation over *known* word tokens, with strict validation.

    Uses token IDs instead of prepare_text to avoid silently discarding
    unsupported graphemes. Keeps original word IDs and absolute timestamps.
    """
    from ctc_segmentation import (
        CtcSegmentationParameters, prepare_token_list,
        ctc_segmentation, determine_utterance_segments)
    if diagnostic is not None:
        diagnostic.update({"status":"started", "words_requested":len(words), "emission_frames":len(log_probs)})
    if log_probs.ndim != 2 or not len(log_probs) or duration <= 0:
        if diagnostic is not None: diagnostic["status"]="invalid_emissions"
        return []
    if len(words) == 0:
        return []
    blank_id = tokenizer.pad_token_id
    if blank_id is None or not 0 <= blank_id < log_probs.shape[1]:
        raise ValueError("CTC tokenizer blank/pad ID incompatible with emissions")
    unknown_id = tokenizer.unk_token_id
    tokens = []
    for word in words:
        # Prefer the native Portuguese spelling when supported. Only remove
        # diacritics when the checkpoint cannot encode the original graphemes.
        raw = "".join(c for c in word["text"].casefold() if c.isalnum())
        alternatives = list(dict.fromkeys((raw, normalize_ctc(word["text"]))))
        ids = None
        for target in alternatives:
            if not target:
                continue
            possible = tokenizer(target, add_special_tokens=False).input_ids
            if possible and all(
                    i != unknown_id and i != blank_id and
                    0 <= i < log_probs.shape[1] for i in possible):
                ids = possible
                break
        if ids is None:
            if diagnostic is not None:
                diagnostic.update(status="unsupported_token",word_id=word.get("id"),token=word.get("text"))
            return []
        tokens.append(np.asarray(ids, dtype=np.int64))
    # The backtracking implementation indexes config.char_list[token_id].
    # prepare_token_list() itself does not populate char_list. Leaving it
    # None raises TypeError: 'NoneType' object is not subscriptable.
    vocabulary = tokenizer.get_vocab()
    if not isinstance(vocabulary, dict):
        raise ValueError("CTC tokenizer did not provide a vocabulary")
    char_list = [f"<token:{i}>" for i in range(log_probs.shape[1])]
    for symbol, index in vocabulary.items():
        if isinstance(index, int) and 0 <= index < len(char_list):
            char_list[index] = str(symbol)
    config = CtcSegmentationParameters()
    config.char_list = char_list
    config.blank = int(blank_id)
    config.index_duration = float(duration) / len(log_probs)
    if not 1 <= score_window <= 100:
        raise ValueError('CTC score window must be between 1 and 100 frames')
    config.score_min_mean_over_L = score_window
    # CTC segmentation expects log-posterior probabilities.
    matrix, starts = prepare_token_list(config, tokens)
    # ctc-segmentation refuses targets longer than the emission sequence.
    # Report this before calling the decoder, rather than raising an
    # opaque AssertionError: Audio is shorter than text!
    if len(matrix) > len(log_probs):
        if diagnostic is not None:
            diagnostic.update(status="insufficient_frames",
                              required_frames=len(matrix),
                              available_frames=len(log_probs),
                              deficit_frames=len(matrix)-len(log_probs))
        return []
    timing, char_probs, _ = ctc_segmentation(config, log_probs, matrix)
    spans = determine_utterance_segments(
        config, starts, char_probs, timing, [w["text"] for w in words])
    if len(spans) != len(words):
        return []
    result = []
    rejected = {}
    previous = offset
    candidate_records = []
    for word, (start, end, score) in zip(words, spans):
        a = float(start) + offset
        b = float(end) + offset
        confidence = float(score)
        reason = None
        if not all(map(math.isfinite, (a, b, confidence))):
            reason = "nonfinite"
        elif confidence < threshold:
            reason = "low_score"
        elif a < offset - .04 or b > offset + duration + .04:
            reason = "out_of_window"
        elif a >= b or b-a > 5.0:
            reason = "invalid_duration"
        elif a < previous - .04:
            reason = "time_regression"
        candidate_records.append({
            "word_id":word["id"],"text":word["text"],
            "start":round(a,4) if math.isfinite(a) else None,
            "end":round(b,4) if math.isfinite(b) else None,
            "score":round(confidence,4) if math.isfinite(confidence) else None,
            "status":"rejected" if reason else "candidate",
            "rejection_reason":reason})
        if reason:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        result.append(Candidate(word["id"], round(max(a, offset), 3),
                                round(min(b, offset+duration), 3),
                                round(confidence, 4)))
        previous = b
    if diagnostic is not None:
        diagnostic.update(status="ok" if result else "all_rejected",
                          candidates=len(result), rejected=rejected,
                          candidate_records=candidate_records,
                          score_window=score_window,
                          score_threshold=threshold,
                          score_range=[round(min(float(x[2]) for x in spans),3),
                                       round(max(float(x[2]) for x in spans),3)])
    return result


class MMSBackend:
    """Lazy, process-local MMS model; one model load for many short windows."""

    def __init__(self, model_id="facebook/mms-1b-all",
                 language="por", device="cpu", cache_dir=None):
        import torch
        from transformers import AutoProcessor, Wav2Vec2ForCTC
        self.torch = torch
        self.device = device
        self.model_id=model_id
        self.language=language
        self.cache_dir=Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True,exist_ok=True)
        self.processor = AutoProcessor.from_pretrained(
            model_id, target_lang=language)
        self.model = Wav2Vec2ForCTC.from_pretrained(
            model_id, target_lang=language,
            ignore_mismatched_sizes=True).to(device).eval()
        # Match tokenizer vocabulary to the loaded language adapter.
        self.processor.tokenizer.set_target_lang(language)
        if not hasattr(self.model, "load_adapter"):
            raise RuntimeError(
                "Installed Transformers model does not expose load_adapter; "
                "cannot verify the MMS language-adapter loading path.")
        self.model.load_adapter(language)
        from .model_integrity import inspect_mms
        self.integrity = inspect_mms(
            self.model, self.processor.tokenizer,
            model_id=model_id, language=language, adapter_requested=True)

    def infer(self, audio16: np.ndarray) -> np.ndarray:
        torch = self.torch
        samples=np.ascontiguousarray(audio16,dtype=np.float32)
        path=None
        if self.cache_dir:
            from transformers import __version__ as transformers_version
            h=sha256()
            for part in (self.model_id,self.language,transformers_version,
                         str(samples.shape),str(samples.dtype)):
                h.update(part.encode());h.update(b"\\0")
            h.update(samples.tobytes())
            path=self.cache_dir/(h.hexdigest()+".npy")
            if path.is_file():
                try:
                    cached=np.load(path,allow_pickle=False)
                    if cached.ndim==2 and cached.shape[1]==self.model.lm_head.out_features:
                        return cached
                except (OSError,ValueError):
                    pass
        payload = self.processor(samples,sampling_rate=16000,
                                 return_tensors="pt")
        payload = {k: v.to(self.device) for k, v in payload.items()}
        with torch.inference_mode():
            logits = self.model(**payload).logits[0]
            result=torch.log_softmax(logits, dim=-1).cpu().numpy()
        if path is not None:
            # Atomic replacement avoids publishing partial cache artifacts.
            fd,tmp=tempfile.mkstemp(prefix="ctc-",suffix=".npy",
                                    dir=str(self.cache_dir))
            try:
                with os.fdopen(fd,"wb") as out:
                    np.save(out,result,allow_pickle=False)
                os.replace(tmp,path)
            finally:
                if os.path.exists(tmp):os.unlink(tmp)
        return result

    def align(self, audio16, words, start, end, *, threshold=-2.5,
              score_window=2):
        if end - start > 20.0:
            raise ValueError("CTC window larger than 20 seconds")
        self.last_diagnostic = {}
        samples = audio16[max(0, int(start*16000)):int(end*16000)]
        if len(samples) < 3200:
            self.last_diagnostic = {"status":"too_short_audio",
                                    "samples":len(samples)}
            return []
        # The continuous mapping is an audit artifact for the NEXT decoder,
        # not an assertion that legacy prepare_token_list includes spaces.
        from .phrase_tokens import tokenize_phrase,check_ctc_capacity
        phrase_audit=None
        try:
            phrase_audit=tokenize_phrase(words,self.processor.tokenizer,
                                vocabulary_size=self.model.lm_head.out_features)
        except ValueError as exc:
            self.last_diagnostic["phrase_tokenization_warning"]=str(exc)
        emission = self.infer(samples)
        if phrase_audit is not None:
            self.last_diagnostic["continuous_reference"]={
                "status":"verified",
                "text":phrase_audit.text,
                "token_ids":list(phrase_audit.token_ids),
                "token_to_word":list(phrase_audit.token_to_word),
                "diacritic_fallback":phrase_audit.used_diacritic_fallback,
                "capacity":check_ctc_capacity(phrase_audit.token_ids,len(emission))}
        return segment_emissions(emission, words, self.processor.tokenizer,
                                 duration=len(samples)/16000.,
                                 offset=start, threshold=threshold,
                                 score_window=score_window,
                                 diagnostic=self.last_diagnostic)


def recover_ctc_gaps(backend, audio16, words: list[dict],
                     *, start: float, end: float,
                     max_gap_seconds: float = 20.,
                     max_gap_words: int = 40,
                     threshold: float = -2.5) -> dict:
    """Retry missing words together with recognized adjacent words.

    Tiny isolated gaps (often 0.1 s) contain insufficient acoustic context
    for CTC. Add up to one aligned word on either side when time permits.
    Only missing IDs may be filled; neighbor timestamps are never replaced.
    """
    count, attempts, i = 0, [], 0
    while i < len(words):
        if words[i]["start"] is not None:
            i += 1
            continue
        j = i
        while j < len(words) and words[j]["start"] is None:
            j += 1
        before = words[i-1] if i else None
        after = words[j] if j < len(words) else None
        gap_start = float(before["end"]) if before else float(start)
        gap_end = float(after["start"]) if after else float(end)
        crop_start, crop_end = gap_start, gap_end
        first, last = i, j
        if before and before.get("start") is not None:
            crop_start = max(float(start), float(before["start"]))
            first = i-1
        if after and after.get("end") is not None:
            crop_end = min(float(end), float(after["end"]))
            last = j+1
        if crop_end-crop_start > max_gap_seconds or last-first > max_gap_words:
            crop_start, crop_end, first, last = gap_start, gap_end, i, j
        attempt = {
            "first_word_id": words[i]["id"], "count": j-i,
            "gap": [round(gap_start, 3), round(gap_end, 3)],
            "window": [round(crop_start, 3), round(crop_end, 3)],
            "context_words": (i-first)+(last-j)}
        if not (0 <= gap_start < gap_end <= end+.1) or (
                crop_end-crop_start > max_gap_seconds) or last-first > max_gap_words:
            attempt["status"] = "skipped_window"
            attempts.append(attempt)
            i = j
            continue
        if crop_end-crop_start < .25:
            attempt["status"] = "skipped_too_short"
            attempts.append(attempt)
            i = j
            continue
        try:
            candidates = backend.align(audio16, words[first:last],
                                       crop_start, crop_end,
                                       threshold=threshold)
            by_id = {candidate.word_id: candidate for candidate in candidates}
            accepted = 0
            for word in words[i:j]:
                candidate = by_id.get(word["id"])
                if candidate is None:
                    continue
                if (candidate.start < gap_start - .04 or
                    candidate.end > gap_end + .04 or
                    candidate.end <= candidate.start):
                    continue
                word.update(start=candidate.start, end=candidate.end,
                            duration=round(candidate.end-candidate.start, 3))
                word["alignment"] = {
                    "status": "tentative", "confidence": None,
                    "method": candidate.method,
                    "acoustic_score": candidate.score}
                accepted += 1
            count += accepted
            attempt["recovered"] = accepted
            attempt["produced_candidates"] = len(candidates)
            attempt["status"] = "processed" if accepted else "no_accepted_candidates"
            diagnostic = getattr(backend, "last_diagnostic", None)
            if isinstance(diagnostic, dict):
                attempt["ctc_diagnostic"] = dict(diagnostic)
        except Exception as exc:
            attempt["status"] = "error"
            attempt["message"] = str(exc)[:350]
        attempts.append(attempt)
        i = j
    statuses = {}
    for attempt in attempts:
        key = attempt["status"]
        statuses[key] = statuses.get(key, 0) + 1
    return {"recovered": count, "attempts": attempts,
            "attempt_statuses": statuses}
