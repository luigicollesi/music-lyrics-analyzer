"""Exact-token and CTC frame feasibility regressions without model download."""
from types import SimpleNamespace
import numpy as np
import pytest

from music_lyrics_analyzer.phrase_tokens import (
    tokenize_phrase,minimum_ctc_frames,check_ctc_capacity)
from music_lyrics_analyzer.ctc_backend import segment_emissions


class CharacterCTCTokenizer:
    pad_token_id=0
    unk_token_id=9
    word_delimiter_token='|'
    word_delimiter_token_id=8
    vocab={"<pad>":0,"f":1,"o":2,"i":3,"s":4,"ã":5,"a":6,"u":7,
           "|":8,"<unk>":9}

    def get_vocab(self):
        return self.vocab

    def __call__(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return SimpleNamespace(input_ids=[
            self.vocab.get("|" if c==" " else c,9) for c in text])


def sample_words():
    return [{"id":"w00001","text":"Foi"},
            {"id":"w00002","text":"só"},
            {"id":"w00003","text":"uma"}]


def test_continuous_mapping_preserves_repeated_occurrence_ids():
    # Use accents only where known to be present in the fake vocabulary.
    t=CharacterCTCTokenizer()
    p=tokenize_phrase([{"id":"w1","text":"Foi"},
                       {"id":"w2","text":"Foi"}],t)
    assert p.text=="foi foi"
    assert p.token_to_word==("w1","w1","w1",None,"w2","w2","w2")
    assert p.token_ids[3]==8


def test_diacritic_fallback_is_explicit():
    t=CharacterCTCTokenizer()
    p=tokenize_phrase([{"id":"w1","text":"Só"}],t)
    assert p.text=="so"
    assert p.used_diacritic_fallback


def test_unknown_character_is_not_silently_dropped():
    t=CharacterCTCTokenizer()
    with pytest.raises(ValueError,match="cannot be encoded"):
        tokenize_phrase([{"id":"w1","text":"xyz"}],t)


def test_noncompositional_phrase_tokenization_fails_closed():
    class Inconsistent(CharacterCTCTokenizer):
        def __call__(self,text,*,add_special_tokens):
            if " " in text:
                return SimpleNamespace(input_ids=[1,8,1])
            return super().__call__(text,add_special_tokens=add_special_tokens)
    with pytest.raises(ValueError,match="differs"):
        tokenize_phrase([{"id":"w1","text":"Foi"},
                         {"id":"w2","text":"Foi"}],Inconsistent())


def test_ctc_minimum_frames_includes_repeated_adjacent_tokens():
    assert minimum_ctc_frames((1,1,2,2,2))==8
    assert check_ctc_capacity((1,1,2),3)["status"]=="insufficient_frames"
    assert check_ctc_capacity((1,1,2),4)["status"]=="feasible"


def test_legacy_ctc_preflight_blocks_decoder(monkeypatch):
    import sys
    class Config:
        blank=0
        score_min_mean_over_L=2
        index_duration=0
        char_list=None
    def prepare(config,tokens):
        return np.zeros((150,1),dtype=np.int64),[0,149]
    def forbidden(*args):
        raise AssertionError("Decoder must not run when there are too few frames.")
    monkeypatch.setitem(sys.modules,"ctc_segmentation",
        SimpleNamespace(CtcSegmentationParameters=Config,
                        prepare_token_list=prepare,
                        ctc_segmentation=forbidden,
                        determine_utterance_segments=forbidden))
    details={}
    out=segment_emissions(np.zeros((20,10),dtype=np.float32),
                          [{"id":"w1","text":"Foi"}],
                          CharacterCTCTokenizer(),duration=.4,diagnostic=details)
    assert out==[]
    assert details["status"]=="insufficient_frames"
    assert details["required_frames"]==150
    assert details["available_frames"]==20
