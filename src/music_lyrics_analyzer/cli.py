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
from .prosody import analyze_vocal_prosody

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
    if args.aligner=='hybrid' and args.strategy!='hierarchical':
        raise ValueError('--aligner hybrid exige --strategy hierarchical.')
    if args.aligner=='ctc' and not args.section_timings:
        raise ValueError('--aligner ctc exige --section-timings; sem âncoras locais, o modelo pode alinhar um refrão à ocorrência errada.')
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
        elif args.aligner in ('stable-ts','hybrid') and args.strategy=='hierarchical':
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
            if args.aligner == 'hybrid':
                print('4/5 Recuperando lacunas com CTC acústico...',file=sys.stderr)
                from .ctc_backend import MMSBackend, recover_ctc_gaps
                ctc = MMSBackend(model_id=args.ctc_model,
                                 language=args.ctc_language,device=args.device)
                if args.ctc_window_mode == 'phrase':
                    from .phrase_recovery import phrase_ctc_recovery
                    recovery_report=phrase_ctc_recovery(
                        ctc,audio16,words,lines,diagnostic.get('sections',[]),
                        threshold=args.ctc_threshold,
                        score_window=args.ctc_score_window)
                    recovered_total=recovery_report['recovered']
                    diagnostic['ctc_summary']={
                        'recovered':recovered_total,
                        'window_mode':'phrase',
                        'attempts':len(recovery_report['attempts']),
                        'candidate_count':len(recovery_report['candidates']),
                        'resolver_accepted':recovery_report['resolver']['accepted_count']}
                    diagnostic.setdefault('attempts',[]).extend(
                        {'ctc_phrase':item}
                        for item in recovery_report['attempts'])
                    candidate_path=output.with_name(output.stem+'-ctc-candidates.json')
                    candidate_path.write_text(json.dumps({
                        'schema_version':1,
                        'model':args.ctc_model,
                        'language':args.ctc_language,
                        'score_window':args.ctc_score_window,
                        'threshold':args.ctc_threshold,
                        'audio_sha256':file_hash(music),
                        'candidates':recovery_report['candidates'],
                        'resolver':recovery_report['resolver']},
                        ensure_ascii=False,indent=2)+'\\n',encoding='utf-8')
                    print(f'  CTC candidatos: {candidate_path}',file=sys.stderr)
                else:
                    recovered_total = 0
                    ctc_statuses = {}
                    ctc_samples = []
                    for section in diagnostic.get('sections', []):
                        window = section.get('window')
                        if window is None:
                            continue
                        section_words = [w for w in words
                                         if w['section_id']==section['id']]
                        report = recover_ctc_gaps(
                            ctc, audio16, section_words,
                            start=window[0], end=window[1],
                            threshold=args.ctc_threshold)
                        recovered_total += report['recovered']
                        for label, number in report.get('attempt_statuses', {}).items():
                            ctc_statuses[label] = ctc_statuses.get(label, 0) + number
                        for item in report.get('attempts', []):
                            if item.get('status') in ('error','no_accepted_candidates',
                                                      'skipped_window','skipped_too_short'):
                                if len(ctc_samples) < 8:
                                    ctc_samples.append({
                                        'word_id': item.get('first_word_id'),
                                        'status': item.get('status'),
                                        'detail': item.get('message') or
                                                  item.get('ctc_diagnostic'),
                                        'window': item.get('window')})
                        diagnostic.setdefault('attempts', []).append({
                            'section_id': section['id'], 'ctc_recovery':report})
                    print(f'  CTC recuperou {recovered_total} palavras candidatas.',file=sys.stderr)
                    print('  CTC status:', ctc_statuses, file=sys.stderr)
                    for sample in ctc_samples:
                        print('  CTC diagnóstico:', json.dumps(sample,ensure_ascii=False),
                              file=sys.stderr)
                    diagnostic['ctc_summary'] = {
                        'recovered': recovered_total, 'statuses': ctc_statuses,
                        'samples': ctc_samples}
                method='hybrid-stable-ts-ctc'
            remaining = sum(w['start'] is None for w in words)
            if remaining:
                warnings.append(f'{remaining} de {len(words)} palavras permanecem sem alinhamento válido.')
        elif args.aligner=='ctc':
            print('3/4 Alinhando com CTC em janelas de seções manuais...',file=sys.stderr)
            from .ctc_backend import MMSBackend, recover_ctc_gaps
            audio16,_=read_audio(vocal_audio,16000)
            if abs(len(audio16)/16000-duration)>1.:
                raise RuntimeError('Vocal separado tem duração incompatível com a música.')
            manual=json.loads(Path(args.section_timings).read_text(encoding='utf-8'))
            if not isinstance(manual,dict):
                raise ValueError('--section-timings exige objeto JSON.')
            ctc=MMSBackend(model_id=args.ctc_model,
                           language=args.ctc_language,device=args.device)
            diagnostic={'strategy':'ctc','sections':[],'errors':[],'attempts':[]}
            for sec in sections:
                bounds=manual.get(sec['id']) or manual.get(sec['label'])
                if bounds is None:
                    diagnostic['sections'].append({'id':sec['id'],
                                                   'status':'unresolved'})
                    continue
                if (not isinstance(bounds,list) or len(bounds)!=2 or
                    not 0 <= float(bounds[0]) < float(bounds[1]) <= duration+.1):
                    raise ValueError('Seção manual inválida: '+sec['label'])
                if float(bounds[1])-float(bounds[0]) > 20.:
                    raise ValueError(
                        'Seção maior que 20 s no modo CTC isolado: '+sec['label']+
                        '. Use --aligner hybrid para localizar janelas menores.')
                region=[w for w in words if w['section_id']==sec['id']]
                report=recover_ctc_gaps(ctc,audio16,region,
                                        start=float(bounds[0]),end=float(bounds[1]),
                                        threshold=args.ctc_threshold)
                diagnostic['attempts'].append({'section_id':sec['id'],
                                               'ctc_recovery':report})
                diagnostic['sections'].append({
                    'id':sec['id'],'window':bounds,'status':'tentative',
                    'aligned_words':report['recovered'],
                    'total_words':len(region)})
            method='ctc-mms'
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
        prosody_summary=None
        if args.prosody=='pyin':
            print('5/5 Medindo prosódia vocal por palavra (pYIN)...',file=sys.stderr)
            prosody_summary,prosody_warnings=analyze_vocal_prosody(
                voice,sr,words,source=source,fmin=args.pitch_min_hz,
                fmax=args.pitch_max_hz,hop_length=args.pitch_hop_length,
                include_contour=args.pitch_contour)
            warnings+=prosody_warnings
    from .candidate_resolver import timeline_issues
    final_issues=timeline_issues(words,duration=duration)
    if final_issues:
        warnings.append(f'{len(final_issues)} inconsistências temporais na timeline final; revisar diagnostics.timeline_issues.')
    sections=collect_sections(lines,words)
    quality={sec['id']:{'coverage':round(sec['aligned_words']/max(1,sec['total_words']),4),
                          'status':('unresolved' if sec['aligned_words']==0 else
                                    'partial' if sec['aligned_words']<sec['total_words'] else 'aligned')}
             for sec in sections}
    for sec in sections:
        sec.update(quality[sec['id']])
    coverage_counts = {
        'total': len(words),
        'timed': sum(w['start'] is not None for w in words),
        'tentative': sum(w['start'] is not None and
                         w['alignment']['status']=='tentative' for w in words),
        'unresolved': sum(w['start'] is None for w in words),
    }
    coverage_counts['non_tentative'] = (
        coverage_counts['timed'] - coverage_counts['tentative'])
    result={'schema_version':2 if args.prosody=='pyin' else 1,
            'audio':{'path':str(music),'duration_seconds':round(duration,3),
                     'sample_rate':sr,'sha256':file_hash(music)},
            'rhythm':rhythm,'sections':sections,'lines':lines,'words':words,
            'coverage':coverage_counts,
            'analysis':{'aligner':method,'separator':args.separator,'language':args.language,
                        'strength_definition':'relative_acoustic_proxy','strength_source':source,
                        'strategy':args.strategy,
                        'prosody_method':args.prosody,
                        'prosody_source':source if args.prosody=='pyin' else None},
            'prosody':prosody_summary,
            'diagnostics':{'sections':diagnostic.get('sections',[]),
                           'errors':diagnostic.get('errors',[]),
                           'attempts':diagnostic.get('attempts',[]),
                           'ctc_summary':diagnostic.get('ctc_summary'),
                           'timeline_issues':final_issues},
            'warnings':warnings}
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if args.lrc:output.with_suffix('.lrc').write_text(lrc_lines(words),encoding='utf-8')
    if args.report:
        report=output.with_name(output.stem+'-report.json')
        report.write_text(json.dumps({'audio_duration':duration,'bpm':rhythm['bpm'],
            'sections':sections,'diagnostics':result['diagnostics'],'warnings':warnings},
            ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print(f'Relatório: {report}',file=sys.stderr)
    print(f'Arquivo: {output}\\nBPM: {rhythm["bpm"]}\\n'
          f'Com horários: {coverage_counts["timed"]}/{coverage_counts["total"]}; '
          f'não-contextuais: {coverage_counts["non_tentative"]}; '
          f'contextuais tentativas: {coverage_counts["tentative"]}; '
          f'sem horário: {coverage_counts["unresolved"]}')
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
    p.add_argument('--aligner',choices=['stable-ts','hybrid','ctc','none'],default='stable-ts')
    p.add_argument('--ctc-model',default='facebook/mms-1b-all',help='Checkpoint Hugging Face CTC, baixado apenas quando usado.')
    p.add_argument('--ctc-language',default='por',help='Adaptador de idioma MMS (Português: por).')
    p.add_argument('--ctc-window-mode',choices=['gap','phrase'],default='phrase',
                   help='CTC por frase com resolução posterior (padrão) ou lacuna isolada (baseline).')
    p.add_argument('--ctc-score-window',type=int,default=2,
                   help='Largura da janela de score do decoder (frames, 1 a 100).')
    p.add_argument('--ctc-threshold',type=float,default=-2.5,help='Limite de score acústico CTC (log-probabilidade); não é confiança calibrada.')
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
    p.add_argument('--prosody',choices=['none','pyin'],default='none',
                   help='Analisar a altura e a dinâmica da voz em cada palavra alinhada.')
    p.add_argument('--pitch-min-hz',type=float,default=75.,help='Limite inferior de frequência fundamental.')
    p.add_argument('--pitch-max-hz',type=float,default=1000.,help='Limite superior de frequência fundamental.')
    p.add_argument('--pitch-hop-length',type=int,default=256,help='Amostras por frame na análise pYIN.')
    p.add_argument('--pitch-contour',action='store_true',
                   help='Exportar curva de pitch por frame dentro de cada palavra.')
    args=p.parse_args(argv)
    try:return analyze(args)
    except (RuntimeError,FileNotFoundError,ValueError,OSError) as exc:
        print('ERRO:',exc,file=sys.stderr)
        return 1
