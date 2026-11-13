"""Offline ASR localization tests; no acoustic model downloads."""
from music_lyrics_analyzer.core import parse_lyrics
from music_lyrics_analyzer.asr_line_search import propose_missing_line_locations


def build():
    words,lines=parse_lyrics(
        "[Verse]\nEspinhos rasgam minha pele mal consigo enxergar\n"
        "Três feras fecham a trilha não há pra onde escapar")
    for i,w in enumerate(words):
        if w["line_id"]==lines[1]["id"]:
            w.update(start=30.+i*.24,end=30.18+i*.24)
    return words,lines


def test_independent_asr_finds_missing_line_before_next_anchors():
    words,lines=build()
    asr=[{"start":24.4,"end":27.3,
          "text":"Espinhos rasgam minha pele mal consigo enxergar"},
         {"start":30.,"end":32.,"text":"Três feras fecham a trilha"}]
    proposed,issues=propose_missing_line_locations(
        words,lines,asr,duration=40.)
    assert len(proposed)==1
    assert proposed[0]["line_id"]==lines[0]["id"]
    assert proposed[0]["end"]<=30.04
    assert proposed[0]["source"]=="asr_lexical_match"


def test_rejects_wrong_phrase_overlap():
    words,lines=build()
    asr=[{"start":30.2,"end":32.5,
          "text":"Espinhos rasgam minha pele mal consigo enxergar"}]
    plans,issues=propose_missing_line_locations(
        words,lines,asr,duration=40.)
    assert not plans
    assert any(x["reason"]=="no_strong_asr_match" for x in issues)


def test_ambiguous_duplicate_phrase_is_not_selected():
    words,lines=build()
    asr=[{"start":22.,"end":24.,
          "text":"Espinhos rasgam minha pele mal consigo enxergar"},
         {"start":25.,"end":27.,
          "text":"Espinhos rasgam minha pele mal consigo enxergar"}]
    plans,issues=propose_missing_line_locations(
        words,lines,asr,duration=40.)
    assert not plans
    assert any(x["reason"]=="ambiguous_asr_occurrence" for x in issues)


def test_weak_match_does_not_force_ctc():
    words,lines=build()
    asr=[{"start":24.,"end":26.,"text":"algumas palavras de outra música"}]
    plans,_=propose_missing_line_locations(words,lines,asr,duration=40.)
    assert plans==[]


def test_short_reference_requires_more_evidence():
    words,lines=parse_lyrics("[Verse]\nFoi só uma vez")
    plans,issues=propose_missing_line_locations(
        words,lines,[{"start":2.,"end":4.,"text":"Foi só uma vez"}],
        duration=10.)
    assert not plans
    assert issues[0]["reason"]=="reference_too_short"
