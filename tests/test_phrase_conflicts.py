"""Phrase-location conflict detection must abstain rather than relabel."""
from music_lyrics_analyzer.core import parse_lyrics
from music_lyrics_analyzer.phrase_conflicts import (
    assess_phrase_conflicts, quarantine_conflicted_candidates)


def fixture():
    words,lines=parse_lyrics(
        "[Verse]\nEspinhos rasgam minha pele mal consigo enxergar\n"
        "Três feras fecham a trilha não há pra onde escapar")
    second=lines[1]["word_ids"]
    for i,w in enumerate(words):
        if w["id"] in second:
            n=second.index(w["id"])
            w["start"]=26.73+n*.27
            w["end"]=w["start"]+.22
    return words,lines


def test_unanchored_line_overlaps_next_anchored_line():
    words,lines=fixture()
    candidates=[{"word_id":w["id"],"start":26.85+i*.32,
                 "end":27.1+i*.32,"status":"candidate","score":-1.}
                for i,w in enumerate(words[:7])]
    audited,report=quarantine_conflicted_candidates(words,lines,candidates)
    assert lines[0]["id"] in report["blocked_line_ids"]
    assert any(x["rejection_reason"]=="phrase_location_conflict"
               for x in audited)
    assert all(x["status"]=="rejected" for x in audited)


def test_nonoverlapping_phrase_is_left_untouched():
    words,lines=fixture()
    candidates=[{"word_id":w["id"],"start":23.+i*.35,
                 "end":23.25+i*.35,"status":"candidate"}
                for i,w in enumerate(words[:7])]
    audited,report=quarantine_conflicted_candidates(words,lines,candidates)
    assert not report["blocked_line_ids"]
    assert all(x["status"]=="candidate" for x in audited)


def test_existing_rejections_are_preserved():
    words,lines=fixture()
    candidate={"word_id":words[0]["id"],"start":27.,
               "end":27.5,"status":"rejected","rejection_reason":"low_score"}
    result,report=quarantine_conflicted_candidates(words,lines,[candidate])
    assert result[0]["rejection_reason"]=="low_score"


def test_line_with_existing_anchor_is_not_quarantined_wholesale():
    words,lines=fixture()
    words[0].update(start=23.3,end=23.6)
    candidates=[{"word_id":words[1]["id"],"start":27.1,
                 "end":27.4,"status":"candidate"}]
    result,report=quarantine_conflicted_candidates(words,lines,candidates)
    assert not report["blocked_line_ids"]
    assert result[0]["status"]=="candidate"


def test_review_mode_does_not_change_original_candidates():
    words,lines=fixture()
    candidates=[{"word_id":words[0]["id"],"start":27.,
                 "end":27.5,"status":"candidate"}]
    original=dict(candidates[0])
    revised,_=quarantine_conflicted_candidates(words,lines,candidates)
    assert candidates[0]==original
    assert revised[0] is not candidates[0]


def test_invalid_threshold_fails():
    words,lines=fixture()
    import pytest
    with pytest.raises(ValueError,match="overlap_ratio"):
        assess_phrase_conflicts(words,lines,[],overlap_ratio=0.)
