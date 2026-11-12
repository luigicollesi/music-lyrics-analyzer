"""Tests for isolated section alignment. No model downloads."""
import numpy as np
from music_lyrics_analyzer.core import parse_lyrics, collect_sections
from music_lyrics_analyzer.localization import (
    find_anchor_candidates, choose_monotonic_anchors,
    section_windows, validate_word_timeline)
from music_lyrics_analyzer.hierarchical import hierarchical_align

LYRICS = """## Chorus (abertura)
Foi só uma vez, só uma vez
Eu disse isso da última vez
## Verse 1
Espinhos rasgam minha pele agora
Três feras fecham a trilha
## Chorus (após Verse 1)
Foi só uma vez, só uma vez
Eu disse isso da última vez
## Verse 2
Os portões de Dite se fecham
Ouço o ferro, sinto o medo
## Final Chorus
Foi mais de uma vez
Agora eu posso escolher
"""
ASR = [
 {'text':'Foi só uma vez só uma vez','start':2,'end':4},
 {'text':'Eu disse isso da última vez','start':4,'end':6},
 {'text':'Espinhos rasgam minha pele agora','start':10,'end':13},
 {'text':'Três feras fecham a trilha','start':13,'end':15},
 {'text':'Foi só uma vez só uma vez','start':22,'end':24},
 {'text':'Eu disse isso da última vez','start':24,'end':26},
 {'text':'Os portões de Dite se fecham','start':31,'end':34},
 {'text':'Ouço o ferro sinto o medo','start':34,'end':36},
 {'text':'Foi mais de uma vez','start':43,'end':45},
 {'text':'Agora eu posso escolher','start':45,'end':48}
]

class Word:
    def __init__(self,text,start,end):
        self.word,self.start,self.end=text,start,end
        self.probability=.95

class Seg:
    def __init__(self,words):
        self.words=words

class Model:
    def align(self,audio,text,language='pt',original_split=True):
        class Result:pass
        result=Result()
        raw=text.split()
        duration=len(audio)/16000.
        words=[Word(t,(i+.1)*duration/(len(raw)+1),(i+.7)*duration/(len(raw)+1))
               for i,t in enumerate(raw)]
        result.segments=[Seg(words)]
        return result

def test_repeated_choruses_choose_unique_anchors():
    words,lines=parse_lyrics(LYRICS)
    sections=collect_sections(lines,words)
    candidates=find_anchor_candidates(lines,ASR)
    anchors=choose_monotonic_anchors(sections,candidates)
    assert len(anchors)==5
    assert all(a is not None for a in anchors)
    assert anchors[0]['start']==2
    assert anchors[2]['start']==22
    assert anchors[4]['start']==43
    assert all(w['window'] is not None for w in section_windows(sections,anchors,50))

def test_missing_section_not_fabricated():
    words,lines=parse_lyrics(LYRICS)
    sections=collect_sections(lines,words)
    asr=[x for x in ASR if not 31<=x['start']<40]
    report=hierarchical_align(Model(),np.zeros(50*16000,dtype=np.float32),
        words,lines,language='pt',duration=50,asr_segments=asr)
    assert report['sections'][3]['status']=='unresolved'
    assert all(w['start'] is None for w in words if w['section_id']==sections[3]['id'])

def test_manual_sections_are_absolute():
    words,lines=parse_lyrics(LYRICS)
    sections=collect_sections(lines,words)
    manual={section['id']:[i*10.,i*10.+8] for i,section in enumerate(sections)}
    report=hierarchical_align(Model(),np.zeros(50*16000,dtype=np.float32),
        words,lines,duration=50,section_timings=manual)
    assert len(report['sections'])==5
    assert all(s['coverage']>.6 for s in report['sections'])
    for index,section in enumerate(sections):
        values=[w for w in words if w['section_id']==section['id']]
        assert all(index*10<=w['start']<index*10+8 for w in values if w['start'] is not None)

def test_rejects_time_regression():
    words,_=parse_lyrics('## Verse 1\num dois três')
    for w,start in zip(words,[1.,.5,2.]):
        w.update(start=start,end=start+.2,duration=.2)
    issues=validate_word_timeline(words)
    assert [p['code'] for p in issues]==['TIME_REGRESSION']
    assert words[1]['start'] is None
    assert words[2]['start']==2.

def test_no_audio_anchor_means_no_section_anchor():
    words,lines=parse_lyrics(LYRICS)
    asr=[{'text':'Apenas música instrumental sem voz','start':2,'end':4}]
    assert all(not v for v in find_anchor_candidates(lines,asr,threshold=.70).values())
