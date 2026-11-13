"""Contract tests for the external CTC decoder, without model downloads."""
import sys
import types
import numpy as np
import pytest

from music_lyrics_analyzer.ctc_backend import segment_emissions


class FakeTokenizer:
    pad_token_id=0
    unk_token_id=9

    def get_vocab(self):
        return {"<pad>":0,"h":1,"o":2,"j":3,"e":4,"m":5,"a":6,"n":7,"ã":8,"<unk>":9}

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        dictionary={"h":1,"o":2,"j":3,"e":4,"m":5,"a":6,"n":7,"ã":8}
        return types.SimpleNamespace(input_ids=[dictionary.get(c,9) for c in text])


def fake_package(monkeypatch, spans):
    class Config:
        blank=0
        index_duration=None
        score_min_mean_over_L=None

    def prepare(config, tokens):
        assert config.blank == 0
        assert config.char_list[1] == 'h'
        assert config.char_list[8] == 'ã'
        assert tokens
        return np.array([[0],[1]]), list(range(len(tokens)+1))

    def segment(config, probs, matrix):
        assert config.index_duration > 0
        return np.arange(len(probs)) * config.index_duration, np.zeros(len(probs)), []

    def resolve(config, boundaries, char_probs, timing, labels):
        assert len(labels)==len(spans)
        return spans

    monkeypatch.setitem(sys.modules,"ctc_segmentation",
        types.SimpleNamespace(CtcSegmentationParameters=Config,
                              prepare_token_list=prepare,
                              ctc_segmentation=segment,
                              determine_utterance_segments=resolve))


def test_returns_absolute_word_intervals(monkeypatch):
    fake_package(monkeypatch,[(.2,.5,-.4),(.6,.9,-1.2)])
    words=[{"id":"w1","text":"Hoje"},{"id":"w2","text":"mãe"}]
    candidates=segment_emissions(np.zeros((40,10)),words,FakeTokenizer(),
                                 duration=2.,offset=10.)
    assert [(x.word_id,x.start,x.end) for x in candidates]==[
        ("w1",10.2,10.5),("w2",10.6,10.9)]


def test_discards_bad_acoustic_scores(monkeypatch):
    fake_package(monkeypatch,[(.2,.5,-5.)])
    words=[{"id":"w1","text":"Hoje"}]
    assert segment_emissions(np.zeros((40,10)),words,FakeTokenizer(),
                             duration=2.,threshold=-2.5)==[]


def test_does_not_silently_drop_unsupported_symbols(monkeypatch):
    fake_package(monkeypatch,[(.2,.5,-.1)])
    words=[{"id":"w1","text":"xyz"}]
    assert segment_emissions(np.zeros((40,10)),words,FakeTokenizer(),
                             duration=2.)==[]


def test_rejects_wrong_blank_symbol(monkeypatch):
    fake_package(monkeypatch,[(.2,.5,-.1)])
    t=FakeTokenizer()
    t.pad_token_id=99
    with pytest.raises(ValueError):
        segment_emissions(np.zeros((40,10)),
                          [{"id":"w1","text":"Hoje"}],t,duration=2.)
