"""Regression tests for phrase planning, deferred selection and export."""
import numpy as np

from music_lyrics_analyzer.core import parse_lyrics
from music_lyrics_analyzer.phrase_windows import plan_phrase_windows
from music_lyrics_analyzer.candidate_resolver import (
    resolve_candidates, commit_resolved, timeline_issues)
from music_lyrics_analyzer.phrase_recovery import phrase_ctc_recovery


def structure():
    words,lines=parse_lyrics("[Verse]\nUm dois três quatro\nCinco seis sete oito")
    for idx,at in ((1,1.2),(2,1.5),(3,1.8),(4,3.0),(5,3.3),(6,3.6),(7,3.9)):
        words[idx].update(start=at,end=at+.18)
    return words,lines


def test_phrase_window_includes_missing_first_word():
    words,lines=structure()
    windows,issues=plan_phrase_windows(words,lines,section_start=.5,section_end=5)
    assert len(windows)==1
    assert windows[0].unresolved_ids==(words[0]["id"],)
    assert windows[0].start<words[1]["start"]
    assert not issues


def test_candidate_resolution_preserves_existing_anchors():
    words,lines=structure()
    original=words[1]["start"]
    proposed=[{"word_id":words[0]["id"],"start":.84,"end":1.09,
               "score":-1.2,"status":"candidate"}]
    result=resolve_candidates(words,proposed)
    assert result["accepted_count"]==1
    assert words[0]["start"] is None
    assert commit_resolved(words,result)==1
    assert words[1]["start"]==original
    assert words[0]["alignment"]["status"]=="tentative"


def test_rejected_low_score_is_visible_but_not_selected():
    words,_=structure()
    proposed=[{"word_id":words[0]["id"],"start":.84,"end":1.09,
               "score":-6,"status":"rejected"}]
    result=resolve_candidates(words,proposed)
    assert result["accepted_count"]==0


def test_detector_catches_global_start_regression():
    words,_=structure()
    words[0].update(start=1.4,end=1.6)
    issues=timeline_issues(words,duration=10)
    assert any(p["reason"]=="start_regression" for p in issues)


class FakeCTC:
    def align(self, audio16, selected, start, end, *, threshold, score_window):
        self.last_diagnostic={"status":"ok","rejected":{},
                              "candidate_records":[{
                                  "word_id":w["id"],
                                  "text":w["text"],"start":start+.05,
                                  "end":start+.20,"score":-.1,
                                  "status":"candidate","rejection_reason":None}
                                  for w in selected]}
        return []


def test_phrase_ctc_collects_then_commits_and_records():
    words,lines=structure()
    report=phrase_ctc_recovery(
        FakeCTC(),np.zeros(5*16000),words,lines,
        [{"id":words[0]["section_id"],"window":[.5,5.]}])
    assert report["candidates"]
    assert report["attempts"]
    assert report["recovered"]==1


def test_zero_duration_section_fails_open():
    words,lines=structure()
    windows,issues=plan_phrase_windows(words,lines,section_start=1,section_end=1)
    assert windows==[]
    assert issues[0]["reason"]=="invalid_section_window"


def test_fully_missing_line_is_not_forced_into_subsecond_crop():
    words,lines=parse_lyrics(
        "[Verse]\nEspinhos rasgam minha pele mal consigo enxergar\n"
        "Três feras fecham a trilha")
    # First line has 7 unresolved words. The first verified word in
    # the next line begins at 26.73s.
    words[7].update(start=26.73,end=27.02)
    windows,issues=plan_phrase_windows(words,lines,
                                      section_start=23.27,section_end=31.)
    assert not issues
    first=next(w for w in windows if w.id=="phrase-"+lines[0]["id"])
    assert first.end-first.start>=2.5
    assert first.start>=23.27


def test_missing_line_reports_insufficient_section_boundaries():
    words,lines=parse_lyrics(
        "[Verse]\nEspinhos rasgam minha pele mal consigo enxergar")
    windows,issues=plan_phrase_windows(words,lines,
                                      section_start=26.73,section_end=27.57)
    assert windows==[]
    assert issues[0]["reason"]=="insufficient_section_context"
