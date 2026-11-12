"""Hierarchical audio-to-lyrics alignment, independent of Markdown and export."""
from __future__ import annotations
import json
import math
from pathlib import Path
import sys
import numpy as np
from .core import apply_alignment, collect_sections
from .localization import (find_anchor_candidates,choose_monotonic_anchors,
                           section_windows,validate_word_timeline)

def _words(result):
    candidates=[]
    if result is None:return candidates
    for seg in (getattr(result,'segments',None) or []):
        for word in (getattr(seg,'words',None) or []):
            candidates.append({'text':word.word,
                'start':getattr(word,'start',None),'end':getattr(word,'end',None),
                'confidence':getattr(word,'probability',None)})
    return candidates

def recognition_windows(model, audio16, *, language='pt', chunk_seconds=30.,
                        print_progress=True):
    """ASR is an anchor hint, not the authoritative lyric transcription."""
    sr=16000
    chunk_len=int(chunk_seconds*sr)
    if chunk_len <= 0:raise ValueError('chunk_seconds must be positive')
    hop=max(sr,chunk_len-int(2*sr))
    segments=[]
    for offset in range(0,len(audio16),hop):
        end=min(len(audio16),offset+chunk_len)
        audio=audio16[offset:end]
        if len(audio)<sr:break
        start_seconds=offset/sr
        if print_progress:
            print(f'  ASR {start_seconds:.1f}s–{end/sr:.1f}s',file=sys.stderr)
        result=model.transcribe(audio,language=language,
                                condition_on_previous_text=False,
                                verbose=None,word_timestamps=False)
        for seg in (getattr(result,'segments',None) or []):
            rel_start=float(seg.start);rel_end=float(seg.end)
            mid=start_seconds+(rel_start+rel_end)/2
            # Keep the center of each overlap only once.
            if offset>0 and mid < start_seconds+1.:continue
            if end<len(audio16) and mid > end/sr-1.:continue
            segments.append({'start':round(start_seconds+rel_start,3),
                             'end':round(start_seconds+rel_end,3),
                             'text':str(seg.text).strip()})
        if end==len(audio16):break
    segments.sort(key=lambda item:item['start'])
    return segments

def _rebase(result,offset):
    for word in result:
        if word.get('start') is not None:word['start']=round(float(word['start'])+offset,3)
        if word.get('end') is not None:word['end']=round(float(word['end'])+offset,3)
    return result

def _window_candidates(model,audio16,lines,start,end,language):
    sr=16000
    section=audio16[int(start*sr):int(end*sr)]
    if len(section)<sr:return []
    lyrics='\n'.join(line['text'] for line in lines)
    result=model.align(section,lyrics,language=language,original_split=True)
    return _rebase(_words(result),start)

def _section_result(words,section):
    relevant=[w for w in words if w['section_id']==section['id']]
    aligned=sum(w['start'] is not None for w in relevant)
    return {'id':section['id'],'aligned_words':aligned,'total_words':len(relevant),
            'coverage':round(aligned/max(1,len(relevant)),4)}

def hierarchical_align(model, audio16, words, lines, *, language='pt',
                       duration=None, section_timings=None,
                       asr_segments=None, chunk_seconds=30.,
                       max_window_seconds=100.,
                       recovery=True):
    """Return raw diagnostics. All unmatched words keep null timestamps.

    Manual sections are a test/debug override. ASR without trustworthy anchors
    never causes invented word timestamps.
    """
    if duration is None:duration=len(audio16)/16000.
    sections=collect_sections(lines,words)
    if asr_segments is None and section_timings is None:
        asr_segments=recognition_windows(model,audio16,language=language,
                                         chunk_seconds=chunk_seconds)
    if section_timings is not None:
        anchors=[]
        for section in sections:
            times=section_timings.get(section['id']) or section_timings.get(section['label'])
            if not times:
                anchors.append(None);continue
            start,end=times
            if not (0<=start<end<=duration+.1):
                raise ValueError('Intervalo manual inválido para '+section['label'])
            anchors.append({'start':float(start),'end':float(end),
                            'score':1.0,'source':'manual'})
        windows=[]
        for section,anchor in zip(sections,anchors):
            windows.append({'id':section['id'],'anchor':anchor,
                'window':[anchor['start'],anchor['end']] if anchor else None,
                'status':'provided' if anchor else 'unresolved',
                'reason':None if anchor else 'no_manual_section'})
        candidates={s['id']:[] for s in sections}
    else:
        candidates=find_anchor_candidates(lines,asr_segments or [])
        anchors=choose_monotonic_anchors(sections,candidates)
        windows=section_windows(sections,anchors,duration,
                                max_section_seconds=max_window_seconds)
    errors=[];attempts=[]
    for s,win in zip(sections,windows):
        if win['window'] is None:continue
        start,end=win['window']
        region_lines=[l for l in lines if l['section_id']==s['id']]
        region_words=[w for w in words if w['section_id']==s['id']]
        try:
            proposals=_window_candidates(model,audio16,region_lines,start,end,language)
            warnings=apply_alignment(region_words,proposals,method='stable-ts-local',duration=duration)
            problems=validate_word_timeline(region_words)
            attempts.append({'section_id':s['id'],'window':[start,end],
                             'proposed_words':len(proposals),
                             'warnings':warnings[:8],'problems':problems[:16]})
            errors.extend({'section_id':s['id'],**p} for p in problems)
            # Retry only when the preceding alignment has low coverage.
            coverage=sum(w['start'] is not None for w in region_words)/max(1,len(region_words))
            if recovery and coverage < .75 and end-start > 15:
                # Divide by *known line order* around the midpoint only if
                # transcription provides a reliable local anchor. Do not
                # manufacture equally spaced timing as a recovery fallback.
                win['reason']='low_coverage'
        except Exception as exc:
            errors.append({'section_id':s['id'],'code':'ALIGNER_ERROR',
                           'message':str(exc)[:300]})
            win['reason']='aligner_error'
    summaries=[]
    for section,win in zip(sections,windows):
        summary=_section_result(words,section)
        summary.update({'label':section['label'],'type':section['type'],
                        'anchor_confidence':win['anchor']['score'] if win['anchor'] else None,
                        'window':win['window'],'status':win['status'],
                        'reason':win['reason']})
        summaries.append(summary)
    return {'strategy':'hierarchical','asr_segments':asr_segments or [],
            'candidates':candidates,'sections':summaries,'attempts':attempts,
            'errors':errors}
