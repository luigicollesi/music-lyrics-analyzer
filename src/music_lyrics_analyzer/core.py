"""Lyrics tokenization, sequence alignment, rhythm and acoustic strength."""
from __future__ import annotations
from difflib import SequenceMatcher
from hashlib import sha256
from pathlib import Path
import math
import re
import unicodedata
import numpy as np

TOKEN=re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*",re.UNICODE)
HEADING=re.compile(r"^\\s*\\[([^\\]]+)\\]\\s*$")
MARKDOWN_HEADING=re.compile(r"^\\s*(#{1,6})\\s+(.+?)\\s*#*\\s*$")
SECTION_PATTERN=re.compile(
    r"^(?P<kind>final\\s+chorus|post[- ]?chorus|pre[- ]?chorus|"
    r"chorus|refr[aã]o|verse|verso|bridge|ponte|intro|introduction|"
    r"outro|ending|instrumental|interlude|interl[uú]dio|hook|"
    r"prel[uú]dio|solo|break|coda)(?=\\s|\\d|\\(|$)", re.IGNORECASE)
SECTION_TYPES={'refrão':'chorus','verso':'verse','ponte':'bridge',
               'introduction':'intro','ending':'outro','interlúdio':'interlude','interludio':'interlude'}
NON_LYRICAL=re.compile(r"^\\s*(?:\\*\\*.*\\*\\*|>.*|<!--.*-->|"
                       r"[-*_]{3,}|\\|.*\\|)\\s*$")
 
def section_info(label):
    """Return normalized section type and optional parenthetical descriptor."""
    cleaned=re.sub(r'\\s+', ' ', label.strip())
    hit=SECTION_PATTERN.match(cleaned)
    if not hit: return None
    kind=hit.group('kind').casefold().replace(' ','-')
    kind=SECTION_TYPES.get(kind,kind)
    extra=re.search(r'\\(([^)]+)\\)\\s*$', cleaned)
    return kind,extra.group(1).strip() if extra else None

def parse_lyrics(text: str):
    """Parse Markdown headings and old [Verse] tags without singing metadata."""
    words,lines=[],[]
    current=None
    markdown=False
    blocked=False
    for raw in text.splitlines():
        raw=raw.strip()
        head=MARKDOWN_HEADING.match(raw)
        bracket=HEADING.match(raw)
        if head:
            markdown=True
            depth=len(head.group(1))
            label=head.group(2).strip()
            parsed=section_info(label) if depth>=2 else None
            if parsed is None:
                current=None
                blocked=True
                continue
        elif bracket:
            label=bracket.group(1).strip()
            parsed=section_info(label)
            if parsed is None:
                current=None
                blocked=True
                continue
        else:
            parsed=None
        if parsed is not None:
            kind,detail=parsed
            section_id=f'section_{sum(1 for item in lines if item.get("section_first"))+1:04d}'
            current={'id':section_id,'label':label,'type':kind,'detail':detail,
                     'first':True}
            blocked=False
            continue
        if not raw or NON_LYRICAL.match(raw) or raw.startswith('!['):
            continue
        if blocked or (markdown and current is None):
            continue
        tokens=TOKEN.findall(raw)
        if not tokens: continue
        if current is None:
            current={'id':'section_0001','label':'Unlabeled','type':'unknown',
                     'detail':None,'first':True}
        lid=f'line_{len(lines)+1:04d}'
        lines.append({'id':lid,'section':current['label'],
                      'section_id':current['id'],'section_type':current['type'],
                      'section_first':current['first'],'text':raw,'word_ids':[]})
        current['first']=False
        for token in tokens:
            wid=f'w{len(words)+1:05d}'
            words.append({'id':wid,'line_id':lid,
                          'section_id':current['id'],'section_type':current['type'],
                          'text':token,'start':None,
                          'end':None,'duration':None,'strength':None,'features':None,
                          'alignment':{'status':'unresolved','confidence':None,'method':None}})
            lines[-1]['word_ids'].append(wid)
    if not words: raise ValueError('A letra não contém palavras utilizáveis.')
    return words,lines

def collect_sections(lines,words):
    """Build ordered, occurrence-specific sections and aligned time spans."""
    by_id={w['id']:w for w in words}
    sections=[]
    for line in lines:
        if not sections or sections[-1]['id']!=line['section_id']:
            sections.append({'id':line['section_id'],'label':line['section'],
                             'type':line['section_type'],'line_ids':[],
                             'word_ids':[],'start':None,'end':None,
                             'aligned_words':0,'total_words':0})
        section=sections[-1]
        section['line_ids'].append(line['id'])
        section['word_ids'].extend(line['word_ids'])
    for section in sections:
        values=[by_id[wid] for wid in section['word_ids']]
                if by_id[wid]['start'] is not None]
        section['total_words']=len(section['word_ids'])
        section['aligned_words']=len(values)
        if values:
            section['start']=min(w['start'] for w in values)
            section['end']=max(w['end'] for w in values)
    return sections

def normalize(token):
    return ''.join(c for c in unicodedata.normalize('NFKC',token).casefold()
                   if unicodedata.category(c).startswith(('L','N')))

def apply_alignment(words,candidates,*,method,duration):
    """Match monotonically, keeping unmatched occurrences unresolved."""
    items=[]
    for item in candidates:
        for part in TOKEN.findall(str(item.get('text',''))):
            if normalize(part): items.append((normalize(part),item))
    expected=[normalize(w['text']) for w in words]
    warnings=[]; matched=set()
    for block in SequenceMatcher(None,expected,[i[0] for i in items],autojunk=False).get_matching_blocks():
        for offset in range(block.size):
            i,j=block.a+offset,block.b+offset
            source=items[j][1]
            try: start,end=float(source['start']),float(source['end'])
            except (KeyError,TypeError,ValueError): continue
            if not(math.isfinite(start) and math.isfinite(end) and 0<=start<end<=duration+.05):
                warnings.append(f'Intervalo inválido em {words[i]["id"]}')
                continue
            w=words[i]
            w['start']=round(start,3);w['end']=round(min(duration,end),3)
            w['duration']=round(w['end']-w['start'],3)
            confidence=source.get('confidence',source.get('probability'))
            try: confidence=float(confidence) if confidence is not None else None
            except (ValueError,TypeError): confidence=None
            if confidence is not None and not math.isfinite(confidence): confidence=None
            w['alignment']={'status':'aligned',
                            'confidence':round(min(1,max(0,confidence)),3) if confidence is not None else None,
                            'method':method}
            matched.add(i)
    if len(matched)!=len(words):
        warnings.append(f'{len(words)-len(matched)} de {len(words)} palavras não alinhadas; revisar áudio/letra.')
    return warnings

def beat_analysis(y,sr):
    import librosa
    if len(y)<sr or not np.any(np.abs(y)>1e-5):
        return {'bpm':None,'confidence':0.0,'alternatives':[],'beats':[],'tempo_changes':[]}
    hop=512
    onset=librosa.onset.onset_strength(y=y,sr=sr,hop_length=hop)
    tempo,frames=librosa.beat.beat_track(onset_envelope=onset,sr=sr,hop_length=hop,trim=False)
    bpm=float(np.asarray(tempo).reshape(-1)[0]) if np.asarray(tempo).size else 0.
    times=[round(float(t),3) for t in librosa.frames_to_time(frames,sr=sr,hop_length=hop)
           if 0<=t<=len(y)/sr]
    if bpm<=0 or not math.isfinite(bpm) or len(times)<3:
        return {'bpm':None,'confidence':0.0,'alternatives':[],'beats':times,'tempo_changes':[]}
    intervals=np.diff(times)
    spread=float(np.std(intervals)/max(float(np.median(intervals)),1e-6))
    return {'bpm':round(bpm,2),'confidence':round(float(np.clip(1-spread*2.5,.05,.95)),3),
            'alternatives':[round(bpm/2,2),round(bpm*2,2)],'beats':times,'tempo_changes':[]}

def strength_analysis(y,sr,words,*,source):
    segments=[]
    for w in words:
        if w['start'] is None: continue
        samples=y[int(w['start']*sr):int(w['end']*sr)]
        frame=max(64,int(.025*sr))
        n=len(samples)//frame
        if n<1: continue
        rms=np.sqrt(np.mean(samples[:n*frame].reshape(n,frame).astype(np.float64)**2,axis=1)+1e-12)
        energy=float(np.percentile(rms,75))
        attack=float(np.max(np.maximum(np.diff(rms,prepend=rms[0]),0)))
        segments.append((w,energy,attack))
    if not segments:return ['Não há palavras alinhadas com áudio suficiente para medir intensidade.']
    E=np.array([20*np.log10(e+1e-8) for _,e,_ in segments])
    A=np.array([20*np.log10(a+1e-8) for _,_,a in segments])
    def scale(x):
        lo,hi=np.percentile(x,[10,90])
        return np.full(len(x),50.) if hi-lo<1e-5 else np.clip((x-lo)/(hi-lo)*100,0,100)
    e,a=scale(E),scale(A)
    p=scale(np.array([v-np.median(E[max(0,i-4):min(len(E),i+5)]) for i,v in enumerate(E)]))
    for i,(w,_,_) in enumerate(segments):
        w['strength']=int(round(.55*e[i]+.30*a[i]+.15*p[i]))
        w['features']={'energy':int(round(e[i])),'attack':int(round(a[i])),'prominence':int(round(p[i]))}
    return [] if source=='vocals' else ['Força medida no áudio mixado: instrumentos podem afetar os valores.']

def file_hash(path: Path):
    h=sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def lrc_lines(words):
    out=[]
    for w in words:
        if w['start'] is None:continue
        t=float(w['start']);m=int(t//60)
        out.append(f'[{m:02d}:{t-m*60:05.2f}]{w["text"]}')
    return '\n'.join(out)+'\n'
