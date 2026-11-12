"""Hierarchical audio-to-lyrics alignment, independent of Markdown and export."""
from __future__ import annotations
import json
import math
from pathlib import Path
import sys
import numpy as np
from .core import apply_alignment, collect_sections
from .recovery import recover_with_word_asr
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

def _line_groups(section_lines, asr_segments, start, end, *,
                 max_local_seconds=35.):
    """Return bounded line groups using ASR evidence, without time interpolation."""
    region=[seg for seg in asr_segments
            if seg.get('start',0)>=start-.5 and seg.get('end',0)<=end+.5]
    if not region or not section_lines:
        return [(section_lines,start,end)] if end-start<=max_local_seconds else []
    # Reuse exactly the same candidate matcher for each line rather than
    # trusting textual position within a long verse.
    proxy=[{**line,'section_id':line['id']} for line in section_lines]
    candidates=find_anchor_candidates(proxy,region,threshold=.58,top_n=6)
    chosen=choose_monotonic_anchors([{'id':line['id']} for line in section_lines],
                                    candidates,min_gap=.05)
    anchored=[(i,a) for i,a in enumerate(chosen) if a is not None]
    if not anchored:
        return [(section_lines,start,end)] if end-start<=max_local_seconds else []
    groups=[]
    first_i,first_a=anchored[0]
    if first_i>0 and first_a['start']-start>1:
        stop=first_a['start']-.05
        if stop-start<=max_local_seconds:
            groups.append((section_lines[:first_i],start,stop))
    for k,(i,a) in enumerate(anchored):
        stop_i,stop=(anchored[k+1][0], anchored[k+1][1]['start']-.05) if k+1<len(anchored) else (len(section_lines),end)
        begin=max(start,a['start']-.75)
        stop=min(end,stop)
        if stop<=begin or stop-begin>max_local_seconds: continue
        groups.append((section_lines[i:stop_i],begin,stop))
    return [(gl,round(gs,3),round(ge,3)) for gl,gs,ge in groups if gl]

def _attempt_recovery(model,audio16,words,lines,asr_segments,start,end,language):
    """Retry an unresolved line only when a strong ASR phrase anchors it."""
    missing=[line for line in lines if not any(
        w['start'] is not None for w in words if w['line_id']==line['id'])]
    if not missing or not asr_segments:return []
    local=[seg for seg in asr_segments
           if seg.get('start',0)>=start and seg.get('end',0)<=end]
    proxy=[{**line,'section_id':line['id']} for line in missing]
    matches=find_anchor_candidates(proxy,local,threshold=.78,top_n=3)
    recovered=[]
    for line in missing[:4]:
        options=matches.get(line['id'],[])
        if not options:continue
        candidate=options[0]
        left=max(start,candidate['start']-.6)
        right=min(end,candidate['end']+.6)
        if right-left<.8 or right-left>15:continue
        region_words=[w for w in words if w['line_id']==line['id']]
        try:
            proposed=_window_candidates(model,audio16,[line],left,right,language)
            apply_alignment(region_words,proposed,method='stable-ts-recovery',duration=end)
            validate_word_timeline(region_words)
            if any(w['start'] is not None for w in region_words):
                recovered.append({'line_id':line['id'],
                                  'start':round(left,3),'end':round(right,3),
                                  'confidence':candidate['score']})
        except Exception:
            continue
    return recovered

def hierarchical_align(model, audio16, words, lines, *, language='pt',
                       duration=None, section_timings=None,
                       asr_segments=None, chunk_seconds=30.,
                       max_window_seconds=100.,
                       recovery=True, max_local_seconds=35.):
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
            line_groups=(
                [(region_lines,start,end)]
                if section_timings is not None else
                _line_groups(region_lines,asr_segments or [],start,end,
                             max_local_seconds=max_local_seconds))
            if not line_groups:
                win['reason']='no_safe_local_window'
            for group_lines,gs,ge in line_groups:
                group_ids={line['id'] for line in group_lines}
                group_words=[w for w in region_words if w['line_id'] in group_ids]
                proposed=_window_candidates(model,audio16,group_lines,gs,ge,language)
                warnings=apply_alignment(group_words,proposed,
                                         method='stable-ts-local',duration=duration)
                problems=validate_word_timeline(group_words)
                attempts.append({'section_id':s['id'],
                    'line_ids':[line['id'] for line in group_lines],
                    'window':[gs,ge],'proposed_words':len(proposed),
                    'warnings':warnings[:8],'problems':problems[:16]})
                errors.extend({'section_id':s['id'],**p} for p in problems)
                if recovery and any(w['start'] is None for w in group_words):
                    try:
                        recovered = recover_with_word_asr(
                            model, audio16, group_words, gs, ge, language)
                        attempts.append({'section_id':s['id'],
                            'line_ids':[line['id'] for line in group_lines],
                            'word_asr_recovery':recovered})
                    except Exception as exc:
                        # Optional recovery must never destroy the first pass.
                        errors.append({'section_id':s['id'],
                            'code':'WORD_ASR_RECOVERY_ERROR',
                            'message':str(exc)[:300]})
            # Check that separately processed line groups also agree with each
            # other in the section's absolute time coordinate.
            cross_problems=validate_word_timeline(region_words)
            errors.extend({'section_id':s['id'],**p} for p in cross_problems)
            coverage=sum(w['start'] is not None for w in region_words)/max(1,len(region_words))
            if recovery and coverage<.75 and asr_segments is not None:
                restored=_attempt_recovery(model,audio16,region_words,region_lines,
                                           asr_segments,start,end,language)
                if restored:
                    attempts.append({'section_id':s['id'],'recovery':restored})
            if coverage<.75 and win['reason'] is None:
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
