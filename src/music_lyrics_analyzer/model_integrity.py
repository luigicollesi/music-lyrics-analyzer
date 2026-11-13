"""Audit MMS Portuguese adapter/tokenizer integration without acoustic claims.

These checks validate structural compatibility only. Real transcription tests
are required to establish that language-specific weights work on speech/canto.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ModelIntegrity:
    model_id: str
    language: str
    output_size: int
    vocabulary_size: int
    blank_id: int
    unknown_id: int | None
    adapter_requested: bool
    status: str
    warnings: tuple[str, ...]

    def to_dict(self):
        return asdict(self)


def inspect_mms(model, tokenizer, *, model_id: str, language: str,
                adapter_requested: bool) -> ModelIntegrity:
    """Validate output topology, vocabulary indices, and special tokens.

    Cannot prove trained adapter weights were restored; a deterministic
    decoding check against known Portuguese speech is a separate gate.
    """
    vocab = tokenizer.get_vocab()
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError("MMS tokenizer vocabulary is missing or invalid")
    if any(not isinstance(v, int) or v < 0 for v in vocab.values()):
        raise ValueError("MMS tokenizer has invalid token indices")
    head = getattr(model, "lm_head", None)
    output_size = getattr(head, "out_features", None)
    if not isinstance(output_size, int) or output_size <= 0:
        raise ValueError("MMS output head is missing or invalid")
    indices = set(vocab.values())
    if max(indices) >= output_size:
        raise ValueError(
            f"MMS vocabulary ID {max(indices)} exceeds output size {output_size}")
    blank = getattr(tokenizer, "pad_token_id", None)
    if not isinstance(blank, int) or blank not in indices:
        raise ValueError("CTC blank/pad token is not in the language vocabulary")
    unknown = getattr(tokenizer, "unk_token_id", None)
    if unknown is not None and (not isinstance(unknown, int) or
                                unknown >= output_size or unknown < 0):
        raise ValueError("MMS unknown-token ID is incompatible with output head")
    warnings = []
    if len(indices) != output_size:
        warnings.append(
            f"Language vocabulary uses {len(indices)} IDs; "
            f"CTC head has {output_size} output positions.")
    if not adapter_requested:
        warnings.append("Language adapter loading was not requested.")
    warnings.append(
        "Structural checks do not verify trained weights or singing accuracy; "
        "run a known-utterance transcription test before production.")
    return ModelIntegrity(model_id=model_id, language=language,
                          output_size=output_size,vocabulary_size=len(vocab),
                          blank_id=blank,unknown_id=unknown,
                          adapter_requested=adapter_requested,
                          status="structurally_compatible",warnings=tuple(warnings))
