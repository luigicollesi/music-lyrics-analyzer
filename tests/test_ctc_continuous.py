"""Deterministic continuous CTC paths: repeats, blanks and lyric identity."""
import numpy as np
import pytest

from music_lyrics_analyzer.ctc_continuous import (
    align_ctc_path,token_spans_from_path,word_candidates_from_spans,
    align_continuous_emissions)
from music_lyrics_analyzer.phrase_tokens import PhraseTokens


def emission(symbols, vocab=6):
    result=np.full((len(symbols),vocab),-10.,dtype=np.float32)
    for t,symbol in enumerate(symbols):
        result[t,symbol]=0.
    return result


def test_viterbi_exact_repeated_tokens_require_blank():
    probs=emission([0,1,1,0,1,2,0])
    path,score=align_ctc_path(probs,[1,1,2],blank_id=0)
    assert [int(x) for x in path]==[0,1,1,2,3,5,6]
    assert abs(score)<1e-6


def test_viterbi_rejects_impossible_target():
    with pytest.raises(ValueError,match="insufficient_frames"):
        align_ctc_path(emission([1,1]),[1,1],blank_id=0)


def test_word_spans_ignore_blank_and_delimiter_frames():
    # "a a" with tokenizer delimiter 3; two different occurrence IDs.
    data=emission([0,1,1,0,3,0,1,1,0])
    target=PhraseTokens("a a",(1,3,1),
        ("w1",None,"w2"),("w1","w2"),False)
    path,_=align_ctc_path(data,target.token_ids,blank_id=0)
    spans=token_spans_from_path(data,target,path)
    candidates=word_candidates_from_spans(
        spans,frame_duration=.01,offset=15.,threshold=-2.5)
    assert [(c["word_id"],c["start"],c["end"]) for c in candidates]==[
        ("w1",15.01,15.03),("w2",15.06,15.08)]
    assert all(c["status"]=="candidate" for c in candidates)


def test_free_blank_frames_are_not_added_to_words():
    data=emission([0,1,0,0,0,2,0])
    target=PhraseTokens("ab",(1,2),("w1","w1"),("w1",),False)
    path,_=align_ctc_path(data,target.token_ids,blank_id=0)
    spans=token_spans_from_path(data,target,path)
    candidate=word_candidates_from_spans(
        spans,frame_duration=.01,offset=0.)[0]
    assert candidate["start"]==.01
    assert candidate["end"]==.06
    # The gap is part of the enclosing interval but not labeled as
    # phonetic evidence. Later blank refinement may change boundaries.
    assert spans[0].end_frame==2
    assert spans[1].start_frame==5


def test_reject_nonfinite_emissions_and_blank_targets():
    with pytest.raises(ValueError,match="blank"):
        align_ctc_path(emission([0,1]),[0],blank_id=0)
    bad=emission([0,1])
    bad[0,0]=np.nan
    with pytest.raises(ValueError,match="Nonfinite"):
        align_ctc_path(bad,[1],blank_id=0)


def test_continuous_segment_records_rejections_but_not_confidence(monkeypatch):
    from music_lyrics_analyzer import ctc_continuous as module
    tokens=PhraseTokens("ab",(1,2),("w1","w1"),("w1",),False)
    monkeypatch.setattr(module,"tokenize_phrase",lambda *args,**kwargs:tokens)
    data=emission([0,1,2,0])
    diagnostic={}
    records=align_continuous_emissions(data,[{"id":"w1","text":"ab"}],
        object(),duration=.08,offset=7.,diagnostic=diagnostic)
    assert diagnostic["status"]=="ok"
    assert diagnostic["word_count"]==1
    assert records[0]["start"]==7.02
    assert records[0]["end"]==7.06
    assert "confidence" not in records[0]


def test_continuous_mode_collects_candidates_without_modifying_words():
    from music_lyrics_analyzer.core import parse_lyrics
    from music_lyrics_analyzer.phrase_recovery import phrase_ctc_recovery
    words,lines=parse_lyrics("[Verse]\nFoi uma vez")
    for idx,time in ((1,1.4),(2,1.8)):
        words[idx].update(start=time,end=time+.2)
    before=[w.get("start") for w in words]

    class Fake:
        def align_continuous(self, audio, selected, start, end, *,
                             threshold, score_window):
            self.last_diagnostic={
                "status":"ok",
                "token_spans":[{"index":0,"word_id":selected[0]["id"],
                               "start_frame":3,"end_frame":9}],
                "candidate_records":[{
                    "word_id":selected[0]["id"],
                    "start":.8,"end":1.1,"score":-.3,
                    "status":"candidate","rejection_reason":None}]}
            return []

    result=phrase_ctc_recovery(
        Fake(),np.zeros(16000*5),words,lines,
        [{"id":words[0]["section_id"],"window":[.2,5.]}],
        mode="continuous",commit=False)
    assert result["committed"] is False
    assert result["recovered"]==0
    assert result["candidates"]
    assert [w.get("start") for w in words]==before
    assert any("token_spans" in a for a in result["attempts"])
