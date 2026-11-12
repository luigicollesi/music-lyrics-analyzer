import numpy as np
from music_lyrics_analyzer.core import parse_lyrics,apply_alignment,beat_analysis,strength_analysis,lrc_lines

def test_lyrics_sections_accents_repetitions():
    words,lines=parse_lyrics('[Chorus]\nFoi só uma vez!\n\n[Verse 1]\nFoi só uma vez.')
    assert [x['text'] for x in words]==['Foi','só','uma','vez']*2
    assert len({x['id'] for x in words})==8
    assert lines[1]['section']=='Verse 1'

def test_alignment_never_invents_skipped_word():
    words,_=parse_lyrics('Eu já não sei')
    warnings=apply_alignment(words,[{'text':'Eu','start':.1,'end':.3},
                                    {'text':'não','start':.8,'end':1.1},
                                    {'text':'sei','start':1.2,'end':1.5}],
                            method='manual',duration=2.)
    assert words[1]['start'] is None
    assert words[2]['start']==.8
    assert len(warnings)==1

def test_invalid_times_unresolved():
    words,_=parse_lyrics('Oi')
    apply_alignment(words,[{'text':'Oi','start':1,'end':40}],method='manual',duration=2.)
    assert words[0]['start'] is None

def test_relative_strength():
    sr=22050
    audio=np.concatenate([np.ones(sr//2,dtype=np.float32)*.03,np.ones(sr//2,dtype=np.float32)*.5])
    words,_=parse_lyrics('fraco FORTE')
    apply_alignment(words,[{'text':'fraco','start':0,'end':.49},
                           {'text':'FORTE','start':.5,'end':.99}],method='manual',duration=1.)
    strength_analysis(audio,sr,words,source='vocals')
    assert words[1]['strength']>words[0]['strength']

def test_silent_bpm():
    assert beat_analysis(np.zeros(22050,dtype=np.float32),22050)['bpm'] is None

def test_lrc():
    words,_=parse_lyrics('Olá mundo')
    apply_alignment(words,[{'text':'Olá','start':2,'end':2.2}],method='manual',duration=5.)
    assert '[00:02.00]Olá' in lrc_lines(words)
    assert 'mundo' not in lrc_lines(words)
