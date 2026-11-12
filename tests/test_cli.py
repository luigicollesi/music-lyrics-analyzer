import json
from pathlib import Path
import subprocess
import sys
import wave
import numpy as np

def test_end_to_end_manual_timestamps(tmp_path):
    repo=Path(__file__).resolve().parents[1];sr=22050
    audio=tmp_path/'music.wav'
    data=(np.sin(2*np.pi*220*np.arange(sr*2)/sr)*.3*32767).astype('<i2')
    with wave.open(str(audio),'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(sr);w.writeframes(data.tobytes())
    lyrics=tmp_path/'lyrics.md';lyrics.write_text('# Inferno — Divina Comédia\n\n**Nota não cantada.**\n\n## Verse 1\nOlá mundo',encoding='utf-8')
    timings=tmp_path/'manual.json';timings.write_text(json.dumps([
      {'text':'Olá','start':.2,'end':.7},{'text':'mundo','start':.8,'end':1.5}]),encoding='utf-8')
    output=tmp_path/'result.json'
    p=subprocess.run([sys.executable,str(repo/'analise.py'),
                      '--music-path',str(audio),'--lyrics-path',str(lyrics),
                      '--word-timings',str(timings),'--output',str(output),'--lrc'],
                     capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    result=json.loads(output.read_text(encoding='utf-8'))
    assert result['schema_version']==1
    assert [w['start'] for w in result['words']]==[.2,.8]
    assert result['words'][0]['strength'] is not None
    assert output.with_suffix('.lrc').exists()
    assert result['sections'][0]['label']=='Verse 1'
    assert result['sections'][0]['type']=='verse'
    assert result['sections'][0]['start']==.2
    assert result['sections'][0]['end']==1.5
    assert '[verse] Verse 1' in p.stdout
    assert result['words'][0]['section_id']==result['sections'][0]['id']


def test_cli_pyin_prosody_with_manual_word_timing(tmp_path):
    repo=Path(__file__).resolve().parents[1]
    sr=16000
    t=np.arange(sr*2)/sr
    audio=tmp_path/'voice.wav'
    signal=(.22*np.sin(2*np.pi*220*t)*32767).astype('<i2')
    with wave.open(str(audio),'wb') as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sr)
        out.writeframes(signal.tobytes())
    lyric=tmp_path/'lyrics.md'
    lyric.write_text('## Verse 1\nVoz constante',encoding='utf-8')
    timings=tmp_path/'words.json'
    timings.write_text(json.dumps([
        {'text':'Voz','start':.2,'end':.8},
        {'text':'constante','start':.95,'end':1.8}
    ]),encoding='utf-8')
    output=tmp_path/'analysis.json'
    command=[sys.executable,str(repo/'analise.py'),
             '--music-path',str(audio),'--lyrics-path',str(lyric),
             '--word-timings',str(timings),'--vocals-path',str(audio),
             '--prosody','pyin','--pitch-contour','--output',str(output)]
    result=subprocess.run(command,capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stderr
    payload=json.loads(output.read_text(encoding='utf-8'))
    assert payload['schema_version']==2
    assert payload['prosody']['analyzed_words']==2
    assert payload['prosody']['pitch_resolved_words']==2
    for word in payload['words']:
        assert abs(word['vocal']['pitch']['mean_hz']-220)<12
        assert len(word['vocal']['pitch']['samples'])>=3
        assert word['vocal']['source']=='vocals'
