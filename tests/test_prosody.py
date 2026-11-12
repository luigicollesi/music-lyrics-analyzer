import math
import numpy as np
import pytest
from music_lyrics_analyzer.core import parse_lyrics, apply_alignment
from music_lyrics_analyzer.prosody import (
    _contour, _intensity, analyze_vocal_prosody, _groups
)


def fixture_words():
    words, _ = parse_lyrics("## Verse 1\nOlá mundo não")
    apply_alignment(words, [
        {"text":"Olá","start":.15,"end":.55},
        {"text":"mundo","start":.65,"end":1.15},
    ], method="manual", duration=2.)
    return words


def fake_pitch(chunk, sr, **kwargs):
    times = np.arange(0, len(chunk)/sr, .025)
    f0 = 190 * 2 ** (times * 2.3 / 12)
    return times, f0, np.ones(len(times),bool), np.ones(len(times))


def test_pitch_trend_rising_falling_stable_and_inconclusive():
    times=np.arange(12)/10
    up=200*2**(4*times/12)
    assert _contour(up,times)[0]=="rising"
    assert _contour(up[::-1],times)[0]=="falling"
    assert _contour(np.full(12,200.),times)[0]=="steady"
    assert _contour(np.array([np.nan,np.nan]),np.array([0.,.1]))==(None,None)


def test_intensity_crescendo_and_decrescendo():
    sr=8000
    t=np.arange(sr)/sr
    up=(.015+.3*t)*np.sin(2*np.pi*220*t)
    down=up[::-1]
    assert _intensity(up,sr)["contour"]=="crescendo"
    assert _intensity(down,sr)["contour"]=="decrescendo"
    assert _intensity(np.zeros(sr),sr)["contour"] is None


def test_word_prosody_and_unaligned_word_is_untouched():
    sr=16000;t=np.arange(sr*2)/sr
    audio=(.2*np.sin(2*np.pi*200*t)).astype("float32")
    words=fixture_words()
    summary,warnings=analyze_vocal_prosody(audio,sr,words,
        source="vocals",pitch_extractor=fake_pitch,include_contour=True)
    assert summary["analyzed_words"]==2
    assert summary["pitch_resolved_words"]==2
    assert summary["unresolved_words"]==1
    assert words[2]["vocal"] is None
    for w in words[:2]:
        assert w["vocal"]["pitch"]["mean_hz"] is not None
        assert w["vocal"]["intensity"]["rms_dbfs"] is not None
        assert w["vocal"]["pitch"]["samples"]
        assert all(w["start"]<=point["time"]<w["end"] for point in w["vocal"]["pitch"]["samples"])
        assert w["vocal"]["delivery"]["shouted"] is None
    assert not warnings


def test_mixture_prosody_marked_low_reliability():
    audio=np.ones(32000,dtype=np.float32)*.01
    words=fixture_words()
    _,warnings=analyze_vocal_prosody(audio,16000,words,
        source="mixture",pitch_extractor=fake_pitch)
    assert words[0]["vocal"]["reliability"]=="low"
    assert any("mixado" in w for w in warnings)


def test_low_voicing_does_not_invent_pitch():
    def unvoiced(chunk,sr,**kwargs):
        t=np.arange(0,len(chunk)/sr,.025)
        return t,np.full(len(t),200.),np.zeros(len(t),bool),np.zeros(len(t))
    words=fixture_words()
    audio=np.zeros(32000,dtype=np.float32)
    summary,_=analyze_vocal_prosody(audio,16000,words,
        source="vocals",pitch_extractor=unvoiced)
    assert summary["pitch_resolved_words"]==0
    assert words[0]["vocal"]["pitch"]["mean_hz"] is None
    assert words[0]["vocal"]["pitch"]["voiced_fraction"]==0.


def test_groups_bounded_and_pitch_params_validated():
    words=fixture_words()
    assert len(_groups(words,max_gap=.05))==2
    with pytest.raises(ValueError):
        analyze_vocal_prosody(np.ones(16000,dtype=np.float32),16000,
                              words,fmin=1200.,fmax=900.)
    with pytest.raises(ValueError):
        analyze_vocal_prosody(np.ones(16000,dtype=np.float32),16000,
                              words,hop_length=0)


def test_real_pyin_for_sustained_tone():
    # Verifies actual librosa pYIN, not the synthetic extractor.
    sr=16000
    t=np.arange(sr)/sr
    audio=(.25*np.sin(2*np.pi*220*t)).astype(np.float32)
    words,_=parse_lyrics("## Verse 1\nvoz")
    apply_alignment(words,[{"text":"voz","start":.1,"end":.9}],method="manual",duration=1.)
    result,_=analyze_vocal_prosody(audio,sr,words,source="vocals")
    assert result["pitch_resolved_words"]==1
    assert abs(words[0]["vocal"]["pitch"]["mean_hz"]-220)<8
