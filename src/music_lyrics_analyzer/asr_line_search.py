"""Conservative ASR-backed placement for completely unaligned lyric lines.

ASR provides *independent location hints*, never definitive word timestamps.
Use distinctive contiguous lexical evidence, neighboring lyric-order limits
and ambiguity rejection before running acoustic CTC in a proposed crop.
"""
from __future__ import annotations

from difflib import SequenceMatcher
from math import isfinite
from .localization import tokens


def _match(reference, observed):
    if len(reference)<3 or len(observed)<3:
        return 0.
    # Highest ordered match across whole short ASR hypotheses. Short common
    # fragments must not pull a complete line to the wrong place.
    ratio=SequenceMatcher(None,reference,observed,autojunk=False).ratio()
    common=sum(b.size for b in SequenceMatcher(
        None,reference,observed,autojunk=False).get_matching_blocks())
    coverage=common/len(reference)
    return round(.65*coverage+.35*ratio,5) if coverage>=.65 else 0.


def propose_missing_line_locations(words,lines,asr_segments,*,duration,
                                   threshold=.82,ambiguity_margin=.08,
                                   min_crop=2.5,max_crop=12.,padding=.65,
                                   max_asr_span=3):
    """Return crop proposals only with distinctive, chronology-safe ASR text.

    A line without any aligned words is searched between its immediate
    neighboring anchored lines (or global song bounds when absent).
    Strong matches that overlap a different anchored line are rejected.
    Similar competing occurrences cause abstention, not arbitrary selection.
    """
    if not 0<threshold<=1 or ambiguity_margin<0:
        raise ValueError("Invalid ASR matching policy.")
    lookup={w["id"]:w for w in words}
    by_line={l["id"]:[lookup[wid] for wid in l["word_ids"]
                      if wid in lookup] for l in lines}
    anchors={}
    for line in lines:
        timed=[w for w in by_line[line["id"]] if
               w.get("start") is not None and w.get("end") is not None]
        if timed:
            anchors[line["id"]]=(min(float(w["start"]) for w in timed),
                                  max(float(w["end"]) for w in timed))
    prepared=[]
    for s in asr_segments or []:
        try:
            a,b=float(s["start"]),float(s["end"])
            tok=tokens(str(s.get("text","")))
        except (KeyError,ValueError,TypeError):
            continue
        if isfinite(a) and isfinite(b) and 0<=a<b<=duration+.1 and tok:
            prepared.append((a,b,tok))
    prepared.sort(key=lambda s:s[0])
    proposals,issues=[],[]
    for idx,line in enumerate(lines):
        lid=line["id"]
        if lid in anchors:
            continue
        ref=tokens(line["text"])
        if len(ref)<4:
            issues.append({"line_id":lid,"reason":"reference_too_short"})
            continue
        # Neighboring anchored lyric lines are order constraints. Do not
        # search in audio assigned to a later or earlier lyric line.
        prev=next((anchors[l["id"]][1] for l in reversed(lines[:idx])
                   if l["id"] in anchors),0.)
        nxt=next((anchors[l["id"]][0] for l in lines[idx+1:]
                  if l["id"] in anchors),duration)
        lo=max(0.,prev-.04)
        hi=min(duration,nxt+.04)
        if hi-lo<min_crop:
            issues.append({"line_id":lid,"reason":"insufficient_unoccupied_audio",
                           "bounds":[round(lo,3),round(hi,3)]})
            continue
        matches=[]
        for i,first in enumerate(prepared):
            if first[0]<lo or first[0]>=hi: continue
            combined=[]
            for j in range(i,min(len(prepared),i+max_asr_span)):
                segment=prepared[j]
                if segment[0]>hi or segment[1]>hi+.1:break
                if j>i and segment[0]-prepared[j-1][1]>1.0:break
                combined.extend(segment[2])
                if len(combined)>max(2*len(ref),40):break
                score=_match(ref,combined)
                if score>=threshold:
                    matches.append((score,first[0],segment[1]))
        if not matches:
            issues.append({"line_id":lid,"reason":"no_strong_asr_match"})
            continue
        matches.sort(key=lambda c:-c[0])
        best=matches[0]
        # Exclude near-duplicate ASR windows around the same utterance.
        distinct=[m for m in matches[1:] if abs(m[1]-best[1])>1.0]
        if distinct and distinct[0][0]>=best[0]-ambiguity_margin:
            issues.append({"line_id":lid,"reason":"ambiguous_asr_occurrence",
                           "score":best[0],"alternatives":len(distinct)})
            continue
        a=max(lo,best[1]-padding)
        b=min(hi,best[2]+padding)
        if b-a<min_crop:
            missing=min_crop-(b-a)
            a=max(lo,a-missing/2)
            b=min(hi,b+(min_crop-(b-a)))
        if b-a<min_crop-1e-3 or b-a>max_crop:
            issues.append({"line_id":lid,"reason":"invalid_asr_crop"})
            continue
        proposals.append({"window_id":f"asr-{lid}",
                          "line_id":lid,"section_id":line["section_id"],
                          "start":round(a,4),"end":round(b,4),
                          "word_ids":[w["id"] for w in by_line[lid]],
                          "anchor_count":0,"asr_score":best[0],
                          "source":"asr_lexical_match"})
    return proposals,issues
