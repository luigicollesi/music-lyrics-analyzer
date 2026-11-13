"""Sliding windows and anchored-only overlap review do not invent labels."""
import numpy as np
import pytest
from music_lyrics_analyzer.core import parse_lyrics
from music_lyrics_analyzer.overlap_windows import sliding_windows,anchored_line_windows
from music_lyrics_analyzer.overlap_recovery import overlap_ctc_review


def test_windows_overlap_by_half():
    windows=sliding_windows(0,20,seconds=8,hop=4)
    assert [(w.start,w.end) for w in windows]==[
        (0,8),(4,12),(8,16),(12,20)]


def test_short_audio_does_not_create_zero_length_window():
    windows=sliding_windows(0,3,seconds=8,hop=4)
    assert len(windows)==1 and windows[0].end==3


def test_invalid_hop_fails():
    with pytest.raises(ValueError):
        sliding_windows(0,10,seconds=8,hop=0)


def sample():
    words,lines=parse_lyrics(
        "[Verse]\nUma frase com âncoras\n"
        "Espinhos rasgam minha pele\n"
        "Outra linha bem cantada")
    for word in words:
        if word["line_id"]==lines[0]["id"]:
            word["start"]=4.8
            word["end"]=5.1
        if word["line_id"]==lines[2]["id"]:
            word["start"]=12.1
            word["end"]=12.4
    return words,lines


def test_unanchored_line_is_skipped_instead_of_forced():
    words,lines=sample()
    plans,skipped=anchored_line_windows(
        words,lines,start=0,end=24,seconds=8,hop=4)
    assert any(x["line_id"]==lines[1]["id"] and
               x["reason"]=="insufficient_independent_anchors"
               for x in skipped)
    assert not any(p["line_id"]==lines[1]["id"] for p in plans)


def test_anchored_line_gets_two_contextual_views():
    words,lines=sample()
    plans,_=anchored_line_windows(words,lines,start=0,end=24,
                                  seconds=8,hop=4)
    first=[p for p in plans if p["line_id"]==lines[0]["id"]]
    assert {(p["start"],p["end"]) for p in first}=={(0,8),(4,12)}


def test_review_never_commits_timestamps():
    words,lines=sample()
    initial=[w.get("start") for w in words]
    class Backend:
        def align_continuous(self,audio,selected,start,end,*,threshold):
            self.last_diagnostic={"status":"ok","candidate_records":[
                {"word_id":selected[0]["id"],
                 "start":start+1,"end":start+1.2,"score":-.2,
                 "status":"candidate","rejection_reason":None}]}
    report=overlap_ctc_review(Backend(),np.zeros(24*16000),
                              words,lines,seconds=8,hop=4)
    assert report["mode"]=="overlap"
    assert not report["committed"]
    assert [w.get("start") for w in words]==initial
    assert report["candidates"]
