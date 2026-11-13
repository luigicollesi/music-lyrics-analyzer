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
            if args.resume and manual_sections is None and asr_segments is None and args.asr_backend!='qwen':
                key=sha256((file_hash(vocal_audio)+args.model+args.language+
                            str(args.asr_chunk_seconds)+args.asr_backend+
                            (args.qwen_model if args.asr_backend=='qwen' else '')+
                            str(args.qwen_chunk_seconds if args.asr_backend=='qwen' else '')+'asr-v03').encode()).hexdigest()[:20]
                cache_file=output.parent/'.cache'/f'asr-{key}.json'
                if cache_file.is_file():
                    asr_segments=json.loads(cache_file.read_text(encoding='utf-8'))
                    print('  Reconhecimento auxiliar recuperado do cache.',file=sys.stderr)
            qwen_segments=[]
            qwen_full=None
            if args.asr_backend=='qwen' and manual_sections is None:
                if args.qwen_mode=='full':
                    qwen_cache=None
                    if args.resume:
                        key=sha256((file_hash(vocal_audio)+args.qwen_model+
                                    args.qwen_device+str(args.qwen_max_new_tokens)+
                                    str(args.qwen_gpu_memory_gb)+str(args.qwen_cpu_memory_gb)+
                                    args.qwen_offload_granularity+
                                    str(args.qwen_resident_weights_gib)+
                                    str(args.qwen_activation_reserve_gib)+
                                    'qwen-full-v2').encode()).hexdigest()[:20]
                        qwen_cache=output.parent/'.cache'/f'qwen-full-{key}.json'
                    if qwen_cache and qwen_cache.is_file():
                        qwen_full=json.loads(qwen_cache.read_text(encoding='utf-8'))
                        print('  Transcrição Qwen integral recuperada do cache.',file=sys.stderr)
                    else:
                        from .qwen_backend import transcribe_qwen_full
                        print('  Transcrevendo música inteira com Qwen3-ASR...',file=sys.stderr)
                        qwen_full,_=transcribe_qwen_full(
                            audio16,model_size=args.qwen_model,
                            device=args.qwen_device,
                            max_new_tokens=args.qwen_max_new_tokens,
                            minimum_vram_gb=args.qwen_min_vram_gb,
                            gpu_memory_gb=args.qwen_gpu_memory_gb,
                            cpu_memory_gb=args.qwen_cpu_memory_gb,
                            memory_report=args.qwen_memory_report,
                            encoder_memory_trace=args.qwen_encoder_memory_trace,
                            offload_granularity=args.qwen_offload_granularity,
                            transfer_report=args.qwen_transfer_report,
                            resident_gpu_weights_gib=args.qwen_resident_weights_gib,
                            activation_reserve_gib=args.qwen_activation_reserve_gib)
                        if qwen_cache:
                            qwen_cache.parent.mkdir(parents=True,exist_ok=True)
                            qwen_cache.write_text(json.dumps(qwen_full,ensure_ascii=False,indent=2),
                                                  encoding='utf-8')
                    diagnostic['qwen_asr']={
                        'model':args.qwen_model,'mode':'full',
                        'role':'full_text_diagnostic_not_timing_anchor',
                        'transcript':qwen_full}
                else:
                    if args.qwen_device in ('hybrid','streamed','resident'):
                        raise ValueError('--qwen-device hybrid/streamed/resident requer --qwen-mode full.')
                    qwen_cache=None
                    if args.resume:
                        key=sha256((file_hash(vocal_audio)+args.qwen_model+
                                    str(args.qwen_chunk_seconds)+'qwen-chunks-v1').encode()).hexdigest()[:20]
                        qwen_cache=output.parent/'.cache'/f'qwen-{key}.json'
                    if qwen_cache and qwen_cache.is_file():
                        qwen_segments=json.loads(qwen_cache.read_text(encoding='utf-8'))
                        print('  Qwen ASR por janelas recuperado do cache.',file=sys.stderr)
                    else:
                        from .qwen_backend import transcribe_qwen_windows
                        print('  Executando Qwen3-ASR por janelas...',file=sys.stderr)
                        qwen_segments,_=transcribe_qwen_windows(
                            audio16,model_size=args.qwen_model,
                            device=args.qwen_device,
                            chunk_seconds=args.qwen_chunk_seconds,
                            hop_seconds=args.qwen_chunk_seconds/2,
                            minimum_vram_gb=args.qwen_min_vram_gb)
                        if qwen_cache:
                            qwen_cache.parent.mkdir(parents=True,exist_ok=True)
                            qwen_cache.write_text(json.dumps(qwen_segments,ensure_ascii=False),
                                                  encoding='utf-8')
                    diagnostic['qwen_asr']={
                        'model':args.qwen_model,'mode':'windowed',
                        'segments':len(qwen_segments),
                        'temporal_precision':'chunk_only',
                        'role':'supplemental_text_not_timing_anchor'}
                if not args.asr_segments:
                    from .hierarchical import recognition_windows
                    asr_segments=recognition_windows(
                        model,audio16,language=args.language,
                        chunk_seconds=args.asr_chunk_seconds)
            qwen_diagnostic=diagnostic.get('qwen_asr')
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
            if qwen_diagnostic is not None:
                diagnostic['qwen_asr']=qwen_diagnostic
                if args.qwen_mode=='windowed':
                    diagnostic['qwen_asr']['examples']=qwen_segments[:12]
            method='stable-ts-hierarchical'
            unresolved=sum(w['start'] is None for w in words)
            if args.aligner == 'hybrid':
                print('4/5 Recuperando lacunas com CTC acústico...',file=sys.stderr)
                from .ctc_backend import MMSBackend, recover_ctc_gaps
                # The 1B MMS checkpoint can exceed 4 GB VRAM. Allow
                # Whisper on CUDA while MMS stays on CPU; when both use CUDA,
                # release the Whisper model before loading MMS.
                ctc_device=(args.device if args.ctc_device=='inherit'
                            else args.ctc_device)
                del model
                if args.device == 'cuda':
                    import gc
                    gc.collect()
                    try:
                        import torch
                        torch.cuda.empty_cache()
                    except (ImportError,RuntimeError):
                        pass
                ctc = MMSBackend(model_id=args.ctc_model,
                                 language=args.ctc_language,device=ctc_device,
                                 cache_dir=(output.parent/'.cache'/'ctc-emissions')
                                 if args.resume else None)
                if args.ctc_window_mode in ('phrase','continuous','overlap'):
                    if args.ctc_window_mode=='overlap':
                        from .overlap_recovery import overlap_ctc_review
                        recovery_report=overlap_ctc_review(
                            ctc,audio16,words,lines,
                            seconds=args.ctc_overlap_seconds,
                            hop=args.ctc_overlap_hop,
                            threshold=args.ctc_threshold,
                            asr_segments=diagnostic.get('asr_segments'))
                    else:
                        from .phrase_recovery import phrase_ctc_recovery
                        recovery_report=phrase_ctc_recovery(
                            ctc,audio16,words,lines,diagnostic.get('sections',[]),
                            threshold=args.ctc_threshold,
                            score_window=args.ctc_score_window,
                            mode=args.ctc_window_mode,
                            commit=args.ctc_window_mode=='phrase')
                    recovered_total=recovery_report['recovered']
                    diagnostic['ctc_summary']={
                        'recovered':recovered_total,
                        'window_mode':args.ctc_window_mode,
                        'experimental_only':args.ctc_window_mode in ('continuous','overlap'),
                        'stability':recovery_report.get('stability',[]),
                        'asr_localization':recovery_report.get('asr_localization'),
                        'attempts':len(recovery_report['attempts']),
                        'candidate_count':len(recovery_report['candidates']),
                        'resolver_accepted':recovery_report['resolver']['accepted_count'],
                        'phrase_conflicts':recovery_report['phrase_conflicts'],
                        'model_integrity':ctc.integrity.to_dict()}
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
                        'mode':args.ctc_window_mode,
                        'stability':recovery_report.get('stability',[]),
                        'asr_localization':recovery_report.get('asr_localization'),
                        'committed':recovery_report['committed'],
                        'audio_sha256':file_hash(music),
                        'model_integrity':ctc.integrity.to_dict(),
                        'candidates':recovery_report['candidates'],
                        'phrase_conflicts':recovery_report['phrase_conflicts'],
                        'resolver':recovery_report['resolver']},
                        ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
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
                        'samples': ctc_samples,
                        'model_integrity':ctc.integrity.to_dict()}
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
                           language=args.ctc_language,
                           device=(args.device if args.ctc_device=='inherit'
                                   else args.ctc_device),
                           cache_dir=(output.parent/'.cache'/'ctc-emissions')
                           if args.resume else None)
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
    if args.lrc:
        timed_count=sum(word['start'] is not None for word in words)
        lrc=lrc_lines(words)
        lrc_count=sum(line.startswith('[') for line in lrc.splitlines())
        if lrc_count!=timed_count:
            raise RuntimeError(f'LRC inconsistente: {lrc_count} entradas vs {timed_count} palavras.')
        output.with_suffix('.lrc').write_text(lrc,encoding='utf-8')
    elif output.with_suffix('.lrc').exists():
        print('AVISO: LRC anterior existe mas não foi regenerado; use --lrc.',file=sys.stderr)
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
    p.add_argument('--ctc-device',choices=['cpu','cuda','inherit'],default='cpu',
                   help='Dispositivo MMS (padrão: cpu; independente do Whisper).')
    p.add_argument('--ctc-window-mode',choices=['gap','phrase','continuous','overlap'],default='phrase',
                   help='gap/phrase: legado; continuous: Viterbi; overlap: revisão com janelas sobrepostas; ambos experimentais.')
    p.add_argument('--ctc-overlap-seconds',type=float,default=8.,
                   help='Duração das janelas acústicas de revisão overlap (padrão: 8s).')
    p.add_argument('--ctc-overlap-hop',type=float,default=4.,
                   help='Avanço entre janelas overlap (padrão: 4s).')
    p.add_argument('--ctc-score-window',type=int,default=2,
                   help='Largura da janela de score do decoder (frames, 1 a 100).')
    p.add_argument('--ctc-threshold',type=float,default=-2.5,help='Limite de score acústico CTC (log-probabilidade); não é confiança calibrada.')
    p.add_argument('--strategy',choices=['hierarchical','legacy'],default='hierarchical',
                   help='Localização hierárquica (padrão) ou alinhamento global anterior.')
    p.add_argument('--asr-backend',choices=['whisper','qwen'],
                   default='whisper',help='Reconhecimento auxiliar para localizar seções.')
    p.add_argument('--qwen-mode',choices=['full','windowed'],default='full',
                   help='Transcrição integral (padrão) ou janelas legadas.')
    p.add_argument('--qwen-max-new-tokens',type=int,default=2048,
                   help='Orçamento de tokens da transcrição integral.')
    p.add_argument('--qwen-model',choices=['0.6b','1.7b'],default='0.6b',
                   help='Modelo Qwen3-ASR opcional; 1.7b exige mais memória.')
    p.add_argument('--qwen-device',choices=['cuda','cpu','auto','hybrid','streamed','resident'],default='cuda',
                   help='Dispositivo Qwen, CUDA explícito por padrão quando selecionado.')
    p.add_argument('--qwen-chunk-seconds',type=float,default=16.,
                   help='Duração da janela de transcrição Qwen (sobreposição 50%%).')
    p.add_argument('--qwen-resident-weights-gib',type=float,default=1.0,
                   help='Orçamento GPU para pesos residentes (modo full/resident).')
    p.add_argument('--qwen-activation-reserve-gib',type=float,default=1.5,
                   help='Reserva minima de VRAM fora do orçamento dos pesos (full/resident).')
    p.add_argument('--qwen-transfer-report',action='store_true',
                   help='Medir chamadas e volume estimado de offloading Accelerate em tempo real.')
    p.add_argument('--qwen-offload-granularity',choices=['tensor','layer'],
                   default='tensor',help='Granularidade experimental do offloading dos pesos do decoder.')
    p.add_argument('--qwen-encoder-memory-trace',action='store_true',
                   help='Rastrear pico de VRAM no encoder de áudio Qwen e em suas camadas.')
    p.add_argument('--qwen-memory-report',action='store_true',
                   help='Registrar checkpoints de consumo CPU/CUDA na inferência full.')
    p.add_argument('--qwen-gpu-memory-gb',type=float,default=2.5,
                   help='Orçamento de VRAM para offloading em modo full/hybrid.')
    p.add_argument('--qwen-cpu-memory-gb',type=float,default=12.,
                   help='Orçamento de RAM para offloading em modo full/hybrid.')
    p.add_argument('--qwen-min-vram-gb',type=float,default=0.,
                   help='Orçamento mínimo de VRAM exigido antes de carregar Qwen.')
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
