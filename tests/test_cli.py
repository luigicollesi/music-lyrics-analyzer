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
    lyrics=tmp_path/'lyrics.txt';lyrics.write_text('[Verse]\nOlá mundo',encoding='utf-8')
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
