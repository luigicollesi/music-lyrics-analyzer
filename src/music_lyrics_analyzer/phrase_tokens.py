"""Reversible, occurrence-aware phrase tokenization for MMS CTC.

This module audits *continuous* tokenizer sequences. It does not replace
the legacy word-segment decoder or claim word-level acoustic accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import groupby
import unicodedata


@dataclass(frozen=True)
class PhraseTokens:
    text: str
    token_ids: tuple[int, ...]
    token_to_word: tuple[str | None, ...]
    word_ids: tuple[str, ...]
    used_diacritic_fallback: bool

    def to_dict(self):
        return asdict(self)


def _normalize_word(text: str, *, strip_accents: bool) -> str:
    text = text.casefold()
    if strip_accents:
        text = ''.join(c for c in unicodedata.normalize('NFKD', text)
                       if unicodedata.category(c) != 'Mn')
    return ''.join(c for c in text if c.isalnum())


def _encode(tokenizer, text):
    result=tokenizer(text, add_special_tokens=False)
    ids=result.input_ids
    if hasattr(ids,'tolist'):
        ids=ids.tolist()
    if ids and isinstance(ids[0], list):
        raise ValueError("Tokenizer unexpectedly returned a batch.")
    return tuple(int(x) for x in ids)


def tokenize_phrase(words, tokenizer, *, vocabulary_size=None) -> PhraseTokens:
    """Tokenize the whole phrase; prove the word mapping by exact re-encoding.

    For character-based Wav2Vec2 CTC tokenizers, individual word tokens plus
    word delimiters should match the encoding of the complete phrase. If not,
    fail closed rather than assigning a misleading token→word mapping.
    """
    if not words:
        raise ValueError("Cannot tokenize an empty phrase.")
    identifiers=[w['id'] for w in words]
    if len(set(identifiers))!=len(identifiers):
        raise ValueError("Phrase contains duplicate word occurrence IDs.")
    blank=getattr(tokenizer,'pad_token_id',None)
    unknown=getattr(tokenizer,'unk_token_id',None)
    vocab=tokenizer.get_vocab()
    if vocabulary_size is None:
        vocabulary_size=max(vocab.values())+1
    delimiter=getattr(tokenizer,'word_delimiter_token_id',None)
    if delimiter is None:
        delimiter=vocab.get(getattr(tokenizer,'word_delimiter_token','|'))
    if delimiter is None:
        raise ValueError("CTC tokenizer lacks an explicit word delimiter.")

    for strip_accents in (False,True):
        strings=[_normalize_word(w['text'],strip_accents=strip_accents)
                 for w in words]
        if not all(strings):
            continue
        phrase=' '.join(strings)
        full_ids=_encode(tokenizer,phrase)
        reconstructed=[]
        owners=[]
        for index,(word,chunk) in enumerate(zip(words,strings)):
            if index:
                reconstructed.append(int(delimiter))
                owners.append(None)
            piece=_encode(tokenizer,chunk)
            reconstructed.extend(piece)
            owners.extend([word['id']]*len(piece))
        valid=lambda xs: bool(xs) and all(
            0<=i<vocabulary_size and i!=blank and i!=unknown for i in xs)
        if not valid(full_ids) or not valid(reconstructed):
            continue
        if full_ids!=tuple(reconstructed):
            # Tokenization depends on cross-word context or special handling.
            # Do not silently claim a reversible token-level mapping.
            raise ValueError("Whole-phrase tokenization differs from the "
                             "word+delimiter sequence; token-to-word mapping "
                             "requires a tokenizer-specific offset adapter.")
        return PhraseTokens(phrase,full_ids,tuple(owners),
                            tuple(identifiers),strip_accents)
    raise ValueError("Portuguese phrase cannot be encoded with the active "
                     "MMS vocabulary, even after diacritic fallback.")


def minimum_ctc_frames(token_ids):
    """Necessary Viterbi CTC frame count for a target token sequence."""
    ids=tuple(token_ids)
    if not ids:
        return 0
    # A blank frame is necessary between adjacent identical tokens because
    # CTC collapse merges contiguous repeats.
    return len(ids)+sum(a==b for a,b in zip(ids,ids[1:]))


def check_ctc_capacity(token_ids, emission_frames):
    """Status is a necessary feasibility test, never evidence of speech."""
    required=minimum_ctc_frames(token_ids)
    return {'status':'feasible' if emission_frames>=required else 'insufficient_frames',
            'required_frames':required,'available_frames':int(emission_frames),
            'deficit_frames':max(0,required-int(emission_frames))}
