"""No-model tests for the CTC backend and conservative recovery."""
import numpy as np

from music_lyrics_analyzer.core import parse_lyrics
from music_lyrics_analyzer.ctc_backend import (
    Candidate, normalize_ctc, recover_ctc_gaps)


def test_portuguese_text_normalizes_for_grapheme_model():
    assert normalize_ctc("Coração!") == "coracao"
    assert normalize_ctc("MÃE") == "mae"


class FakeBackend:
    def __init__(self):
        self.windows = []

    def align(self, audio, words, start, end, *, threshold):
        self.windows.append((start, end, [w["text"] for w in words]))
        return [Candidate(w["id"], start+.1+i*.2,
                          start+.25+i*.2, -.4)
                for i,w in enumerate(words)]


def test_missing_middle_words_use_bounding_anchors():
    words, _ = parse_lyrics("hoje como mingau para comer")
    words[0].update(start=1., end=1.3)
    words[3].update(start=3., end=3.4)
    words[4].update(start=3.5, end=3.8)
    fake = FakeBackend()
    result = recover_ctc_gaps(fake,np.zeros(16000*5),
                              words,start=0,end=5)
    assert result["recovered"] == 2
    assert fake.windows == [(1.3,3.0,["como","mingau"])]
    assert words[1]["alignment"]["method"] == "ctc-mms"
    assert words[1]["alignment"]["status"] == "tentative"
    assert words[0]["start"] == 1.0


def test_long_unanchored_section_is_not_guess_aligned():
    words, _ = parse_lyrics("hoje como mingau")
    fake=FakeBackend()
    report=recover_ctc_gaps(fake,np.zeros(16000),words,
                            start=0,end=100)
    assert report["recovered"] == 0
    assert fake.windows == []


def test_backend_failure_keeps_words_unresolved():
    class Broken:
        def align(self,*args,**kwargs):
            raise RuntimeError("model unavailable")
    words,_=parse_lyrics("hoje como")
    report=recover_ctc_gaps(Broken(),np.zeros(16000),words,
                            start=0,end=2)
    assert report["attempts"][0]["status"] == "error"
    assert all(w["start"] is None for w in words)


def test_ctc_does_not_replace_existing_word():
    words,_=parse_lyrics("hoje como agora")
    words[1].update(start=1.0,end=1.3)
    fake=FakeBackend()
    recover_ctc_gaps(fake,np.zeros(16000*4),words,start=0,end=4)
    assert words[1]["start"] == 1.0
    assert len(fake.windows) == 2
