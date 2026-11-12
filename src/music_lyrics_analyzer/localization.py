"""Pure section-anchor discovery from noisy ASR, preserving song order."""
from __future__ import annotations
from difflib import SequenceMatcher
import re
from .core import normalize, TOKEN

def tokens(value: str) -> list[str]:
    return [normalize(t) for t in TOKEN.findall(value) if normalize(t)]

def similarity(reference: list[str], heard: list[str]) -> float:
    if not reference or not heard:
        return 0.0
    # The ASR may omit or repeat words. Avoid character-only accidental matches.
    order = SequenceMatcher(None, reference, heard, autojunk=False).ratio()
    a, b = set(reference), set(heard)
    overlap = len(a & b) / max(1, len(a | b))
    return .75 * order + .25 * overlap

def find_anchor_candidates(lines, asr_segments, *, threshold=.53, top_n=10):
    """Locate the first distinctive sung line of a section using short ASR windows.

    Return each section's candidates. Multiple chorus matches are retained:
    global order, not local greed, resolves repeated text.
    """
    ids = []
    for line in lines:
        if line['section_id'] not in ids: ids.append(line['section_id'])
    result = {}
    segments = [s for s in asr_segments if tokens(s.get('text',''))
                and s.get('start') is not None and s.get('end') is not None
                and s['end'] > s['start']]
    for section_id in ids:
        section_lines = [line for line in lines if line['section_id']==section_id]
        # First two lines serve as phrase anchors; later lines are less reliable
        # as section START markers. Keep the phrase reasonably long.
        refs = [tokens(l['text']) for l in section_lines[:2]]
        refs = [r for r in refs if len(r) >= 3]
        candidates = []
        for phrase_index, ref in enumerate(refs):
            for i in range(len(segments)):
                combined = []
                for j in range(i, min(i + 5, len(segments))):
                    combined.extend(tokens(segments[j]['text']))
                    if len(combined) > max(2*len(ref)+5, 30): break
                    score = similarity(ref, combined)
                    if score < threshold: continue
                    candidates.append({'section_id':section_id,
                        'start':float(segments[i]['start']),
                        'end':float(segments[j]['end']),
                        'score':round(score,5),'phrase_index':phrase_index,
                        'asr_start_index':i,'asr_end_index':j})
        candidates.sort(key=lambda c:(-c['score'], c['start']))
        # Avoid variants of the same ASR phrase crowding out later choruses.
        unique=[]
        for c in candidates:
            if any(abs(c['start']-x['start']) < 2.0 for x in unique):continue
            unique.append(c)
            if len(unique)>=top_n:break
        result[section_id]=sorted(unique,key=lambda c:c['start'])
    return result

def choose_monotonic_anchors(sections, candidates, *, min_gap=.25):
    """Dynamic programming to maximize anchor confidence without reordering lyrics.

    An unanchored section is allowed and is never assigned a fabricated time.
    """
    # Each candidate is an occurrence in a graph of ordered section positions.
    # A skipped section incurs a penalty; invalid time transitions are prohibited.
    states=[(0.0, [], -1.0)]
    for section in sections:
        next_states=[]
        opts=candidates.get(section['id'], [])
        for score,path,last_end in states:
            next_states.append((score-.31, path+[None], last_end))
            for c in opts:
                if c['start'] < last_end + min_gap: continue
                next_states.append((score + c['score']-.25, path+[c], c['end']))
        next_states.sort(key=lambda x:x[0],reverse=True)
        # Beam is enough for short songs; keep different last_end anchors.
        kept=[];seen=set()
        for x in next_states:
            key=(round(x[2],1),sum(v is not None for v in x[1]))
            if key in seen:continue
            seen.add(key);kept.append(x)
            if len(kept)>=512:break
        states=kept
    return states[0][1] if states else [None]*len(sections)

def section_windows(sections, selected, audio_duration, *, max_section_seconds=100.0):
    """Section-local analysis intervals; missing anchors stay unresolved."""
    output=[]
    for index, (section,anchor) in enumerate(zip(sections,selected)):
        state={'id':section['id'],'anchor':anchor,'window':None,
               'status':'unresolved','reason':'no_anchor' if anchor is None else None}
        if anchor is None:
            output.append(state);continue
        # Next anchor bounds the current section. For the last section,
        # use audio end; long gaps are flagged instead of silently spanning songs.
        future=next((item for item in selected[index+1:] if item is not None),None)
        start=max(0.0,anchor['start']-.75)
        end=min(audio_duration, future['start']-.05 if future else audio_duration)
        if end<=start or end-start>max_section_seconds:
            state['reason']='implausible_window'
        else:
            state['window']=[round(start,3),round(end,3)]
            state['status']='tentative'
        output.append(state)
    return output

def validate_word_timeline(words, *, max_duration=5.0):
    """Invalidate non-monotonic local candidates before they reach downstream VFX."""
    problems=[];previous=-1.0
    for w in words:
        if w['start'] is None:continue
        start,end=w['start'],w['end']
        code=None
        if end<=start: code='ZERO_OR_NEGATIVE_DURATION'
        elif end-start>max_duration: code='IMPLAUSIBLE_DURATION'
        elif start<previous-.04: code='TIME_REGRESSION'
        if code:
            problems.append({'code':code,'word_id':w['id'],'start':start,'end':end})
            w.update(start=None,end=None,duration=None,strength=None,features=None)
            w['alignment']={'status':'unresolved','confidence':None,'method':None}
        else: previous=end
    return problems
