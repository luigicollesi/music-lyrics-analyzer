"""Public CLI: python analise.py --music-path ... --lyrics-path ..."""
from __future__ import annotations
import argparse
import json
from hashlib import sha256
import subprocess
import sys
import tempfile
from pathlib import Path
from .core import parse_lyrics,collect_sections,apply_alignment,beat_analysis,strength_analysis,file_hash,lrc_lines
from .hierarchical import hierarchical_align

def read_audio(path: Path,sr=22050):
    import numpy as np
    try:
        p=subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(path),'-vn',
                          '-ac','1','-ar',str(sr),'-f','f32le','-acodec','pcm_f32le','pipe:1'],
                         capture_output=True,check=False)
    except FileNotFoundError as e:
        raise RuntimeError('FFmpeg não encontrado. Instale ffmpeg e adicione ao PATH.') from e
    if p.returncode:
        raise RuntimeError('FFmpeg falhou: '+p.stderr.decode('utf-8','replace')[-700:])
    y=np.frombuffer(p.stdout,dtype='<f4').copy()
    if not len(y):raise RuntimeError('O áudio está vazio.')
    return y,sr

def stable_ts_align(audio:Path,lyrics:str,language:str,model_name:str,device:str):
    try:import stable_whisper
    except ImportError as e:
        raise RuntimeError('Instale o alinhador com: python -m pip install -e ".[align]"') from e
    model=stable_whisper.load_model(model_name,device=device)
    result=model.align(str(audio),lyrics,language=language,original_split=True)
    if result is None:return []
    return [{'text':word.word,'start':word.start,'end':word.end,
             'confidence':getattr(word,'probability',None)}
            for seg in result.segments for word in (seg.words or [])]

def separate_vocals(audio:Path,temp:Path,device:str):
    p=subprocess.run([sys.executable,'-m','demucs','--two-stems','vocals','-d',device,
                      '-o',str(temp),str(audio)],capture_output=True,text=True)
    if p.returncode:
        raise RuntimeError('Demucs falhou: '+p.stderr[-900:])
    candidates=list(temp.glob('*/'+audio.stem+'/vocals.wav'))
    if not candidates:raise RuntimeError('Demucs não criou vocals.wav.')
    return candidates[0]

def analyze(args):
    music=Path(args.music_path).expanduser().resolve()
    lyrics_file=Path(args.lyrics_path).expanduser().resolve()
    if not music.is_file():raise FileNotFoundError('Áudio inexistente: '+str(music))
    if not lyrics_file.is_file():raise FileNotFoundError('Letra inexistente: '+str(lyrics_file))
    lyric_text=lyrics_file.read_text(encoding='utf-8-sig')
    words,lines=parse_lyrics(lyric_text)
    sections=collect_sections(lines,words)
    print(f'Letra: {len(sections)} seções, {len(lines)} linhas, {len(words)} palavras',file=sys.stderr)
    for index,section in enumerate(sections,1):
        print(f'  {index:02d}. [{section["type"]}] {section["label"]}: {section["total_words"]} palavras',file=sys.stderr)
    aligned_lyrics='\n'.join(line['text'] for line in lines)
    output=Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    y,sr=read_audio(music);duration=len(y)/sr
    print('1/4 Analisando BPM...',file=sys.stderr)
    rhythm=beat_analysis(y,sr)
    warnings=[]
    diagnostic={'strategy':args.strategy,'sections':[],'errors':[]}
    with tempfile.TemporaryDirectory(prefix='music-lyrics-') as folder:
        vocal_audio=music;source='mixture'
        if args.vocals_path:
            vocal_audio=Path(args.vocals_path).expanduser().resolve()
            if not vocal_audio.is_file():raise FileNotFoundError('Vocal não encontrado: '+str(vocal_audio))
            source='vocals'
            print('2/4 Usando vocais pré-separados.',file=sys.stderr)
        elif args.separator=='demucs':
            print('2/4 Separando voz com Demucs...',file=sys.stderr)
            vocal_audio=separate_vocals(music,Path(folder),args.device)
            source='vocals'
        else:print('2/4 Separação vocal desabilitada.',file=sys.stderr)
        if args.word_timings:
            data=json.loads(Path(args.word_timings).read_text(encoding='utf-8'))
            if not isinstance(data,list):raise ValueError('--word-timings exige lista JSON.')
            candidates=data;method='manual'
            warnings+=apply_alignment(words,candidates,method=method,duration=duration)
        elif args.aligner=='stable-ts' and args.strategy=='hierarchical':
            print('3/4 Localizando seções e alinhando palavras...',file=sys.stderr)
            try:
                import stable_whisper
            except ImportError as exc:
                raise RuntimeError('Instale o alinhador: python -m pip install -e ".[align]"') from exc
            model=stable_whisper.load_model(args.model,device=args.device)
            audio16,_=read_audio(vocal_audio,16000)
            if abs(len(audio16)/16000-duration)>1.:
                raise RuntimeError('Vocal separado tem duração incompatível com a música.')
            manual_sections=None
            if args.section_timings:
                manual_sections=json.loads(Path(args.section_timings).read_text(encoding='utf-8'))
                if not isinstance(manual_sections,dict):
                    raise ValueError('--section-timings exige objeto JSON com seções.')
            asr_segments=None
            if args.asr_segments:
                asr_segments=json.loads(Path(args.asr_segments).read_text(encoding='utf-8'))
                if not isinstance(asr_segments,list):
                    raise ValueError('--asr-segments exige lista JSON.')
            cache_file=None
            if args.resume and manual_sections is None and asr_segments is None:
                key=sha256((file_hash(vocal_audio)+args.model+args.language+
                            str(args.asr_chunk_seconds)+'asr-v02').encode()).hexdigest()[:20]
                cache_file=output.parent/'.cache'/f'asr-{key}.json'
                if cache_file.is_file():
                    asr_segments=json.loads(cache_file.read_text(encoding='utf-8'))
                    print('  Reconhecimento auxiliar recuperado do cache.',file=sys.stderr)
            diagnostic=hierarchical_align(
                model,audio16,words,lines,language=args.language,duration=duration,
                section_timings=manual_sections,asr_segments=asr_segments,
                chunk_seconds=args.asr_chunk_seconds,
                max_window_seconds=args.max_section_seconds,
                recovery=args.recovery=='auto',
                max_local_seconds=args.max_local_seconds)
            if cache_file and asr_segments is None and diagnostic['asr_segments']:
                cache_file.parent.mkdir(parents=True,exist_ok=True)
                cache_file.write_text(json.dumps(diagnostic['asr_segments'],
                                                 ensure_ascii=False),encoding='utf-8')
            method='stable-ts-hierarchical'
            unresolved=sum(w['start'] is None for w in words)
            if unresolved:
                warnings.append(f'{unresolved} de {len(words)} palavras permanecem sem alinhamento válido.')
        elif args.aligner=='stable-ts':
            print('3/4 Alinhando letra globalmente (legado)...',file=sys.stderr)
            candidates=stable_ts_align(vocal_audio,aligned_lyrics,args.language,args.model,args.device)
            method='stable-ts-legacy'
            warnings+=apply_alignment(words,candidates,method=method,duration=duration)
        else:
            method='none'
            warnings.append('Alinhador desabilitado: horários não serão inventados.')
        print('4/4 Medindo intensidade acústica...',file=sys.stderr)
        voice,_=read_audio(vocal_audio,sr) if source=='vocals' else (y,sr)
        warnings+=strength_analysis(voice,sr,words,source=source)
    sections=collect_sections(lines,words)
    quality={sec['id']:{'coverage':round(sec['aligned_words']/max(1,sec['total_words']),4),
                          'status':('unresolved' if sec['aligned_words']==0 else
                                    'partial' if sec['aligned_words']<sec['total_words'] else 'aligned')}
             for sec in sections}
    for sec in sections:
        sec.update(quality[sec['id']])
    result={'schema_version':1,
            'audio':{'path':str(music),'duration_seconds':round(duration,3),
                     'sample_rate':sr,'sha256':file_hash(music)},
            'rhythm':rhythm,'sections':sections,'lines':lines,'words':words,
            'analysis':{'aligner':method,'separator':args.separator,'language':args.language,
                        'strength_definition':'relative_acoustic_proxy','strength_source':source,
                        'strategy':args.strategy},
            'diagnostics':{'sections':diagnostic.get('sections',[]),
                           'errors':diagnostic.get('errors',[]),
                           'attempts':diagnostic.get('attempts',[])},
            'warnings':warnings}
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if args.lrc:output.with_suffix('.lrc').write_text(lrc_lines(words),encoding='utf-8')
    if args.report:
        report=output.with_name(output.stem+'-report.json')
        report.write_text(json.dumps({'audio_duration':duration,'bpm':rhythm['bpm'],
            'sections':sections,'diagnostics':result['diagnostics'],'warnings':warnings},
            ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(f'Relatório: {report}',file=sys.stderr)
    print(f'Arquivo: {output}\nBPM: {rhythm["bpm"]}\nAlinhadas: {sum(w["start"] is not None for w in words)}/{len(words)}')
    print('\nSeções identificadas (tempos baseados nas palavras alinhadas):')
    for sec in sections:
        clock=(f'{sec["start"]:.2f}s–{sec["end"]:.2f}s'
               if sec['start'] is not None else 'tempo não identificado')
        print(f'  [{sec["type"]}] {sec["label"]}: {clock} '
              f'({sec["aligned_words"]}/{sec["total_words"]} palavras)')
    for warning in warnings:print('AVISO:',warning,file=sys.stderr)
    return 0

def main(argv=None):
    p=argparse.ArgumentParser(prog='python analise.py',description='Analisa BPM, palavras cantadas e força acústica relativa.')
    p.add_argument('--music-path',required=True,help='Caminho para música WAV, MP3, FLAC, etc.')
    p.add_argument('--lyrics-path',required=True,help='Caminho para letra em Markdown (.md) ou texto simples (.txt), UTF-8.')
    p.add_argument('--output',default='analysis/lyrics-analysis.json',help='Arquivo de saída JSON.')
    p.add_argument('--language',default='pt',help='Idioma da letra (padrão: pt).')
    p.add_argument('--aligner',choices=['stable-ts','none'],default='stable-ts')
    p.add_argument('--strategy',choices=['hierarchical','legacy'],default='hierarchical',
                   help='Localização hierárquica (padrão) ou alinhamento global anterior.')
    p.add_argument('--asr-chunk-seconds',type=float,default=30.,help='Janela do reconhecimento auxiliar em segundos.')
    p.add_argument('--max-section-seconds',type=float,default=150.,help='Duração máxima da janela de alinhamento de uma seção.')
    p.add_argument('--max-local-seconds',type=float,default=35.,help='Duração máxima dos subtrechos de alinhamento.')
    p.add_argument('--recovery',choices=['auto','none'],default='auto')
    p.add_argument('--section-timings',help='JSON opcional de intervalos de seções verificados manualmente.')
    p.add_argument('--asr-segments',help='JSON opcional de reconhecimento auxiliar para testes.')
    p.add_argument('--resume',action='store_true',help='Reutilizar cache do reconhecimento auxiliar.')
    p.add_argument('--report',action='store_true',help='Exportar diagnóstico por seção.')
    p.add_argument('--vocals-path',help='Arquivo de vocais pré-separados, preservando duração e início da música.')
    p.add_argument('--model',default='base',help='Modelo Whisper: tiny, base, small...')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--separator',choices=['none','demucs'],default='none')
    p.add_argument('--word-timings',help='JSON de tempos manuais para testes.')
    p.add_argument('--lrc',action='store_true',help='Gerar também arquivo LRC.')
    args=p.parse_args(argv)
    try:return analyze(args)
    except (RuntimeError,FileNotFoundError,ValueError,OSError) as exc:
        print('ERRO:',exc,file=sys.stderr)
        return 1
