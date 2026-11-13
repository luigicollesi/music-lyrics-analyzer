"""Conservative structural conflict audit for line-level CTC hypotheses.

An acoustic path can be mathematically valid while aligned to a different
lyric line. This module never assigns timings: it flags unsupported phrase
placements for abstention pending independent localization.
"""
from __future__ import annotations
from collections import defaultdict
from math import isfinite


def assess_phrase_conflicts(words, lines, candidates, *, overlap_ratio=.35,
                            tolerance=.04):
    """Inspect candidate coverage of anchored *other* lyric lines.

    Only block an entirely unanchored line when its proposed spans occupy an
    anchored, distinct line's measured interval, beyond overlap_ratio of
    total proposed duration. This is a conservative evidence gate, not an
    oracle: aligned anchors might themselves be inaccurate. The output is
    diagnostic and the caller decides whether to abstain.
    """
    if not 0 < overlap_ratio <= 1:
        raise ValueError("overlap_ratio must be in (0,1]")
    id_to_word={w["id"]:w for w in words}
    id_to_line={w["id"]:w["line_id"] for w in words}
    line_order={line["id"]:i for i,line in enumerate(lines)}
    words_by_line=defaultdict(list)
    for w in words:
        words_by_line[w["line_id"]].append(w)
    anchors={}
    for line_id, group in words_by_line.items():
        timed=[w for w in group if w.get("start") is not None
               and w.get("end") is not None]
        if timed:
            anchors[line_id]=(min(float(w["start"]) for w in timed),
                              max(float(w["end"]) for w in timed))
    by_line=defaultdict(list)
    for c in candidates:
        word_id=c.get("word_id")
        if word_id not in id_to_line: continue
        a,b=c.get("start"),c.get("end")
        if a is None or b is None:continue
        a,b=float(a),float(b)
        if not all(map(isfinite,(a,b))) or not 0<=a<b:continue
        by_line[id_to_line[word_id]].append((a,b,c))
    issues=[]
    blocked=set()
    for line_id, records in by_line.items():
        # If even one word of this line is aligned, individual candidates
        # remain governed by existing local anchor constraints.
        if line_id in anchors:continue
        duration=sum(b-a for a,b,_ in records)
        if duration<=0:continue
        for other, (lo,hi) in anchors.items():
            if other==line_id:continue
            if abs(line_order.get(other,-1000)-line_order.get(line_id,-2000))>2:
                continue
            conflict=sum(max(0.,min(b,hi-tolerance)-max(a,lo+tolerance))
                         for a,b,_ in records)
            ratio=conflict/duration
            if ratio>=overlap_ratio:
                blocked.add(line_id)
                issues.append({
                    "line_id":line_id,"conflicting_line_id":other,
                    "reason":"unanchored_line_overlaps_anchored_neighbor",
                    "overlap_ratio":round(ratio,4),
                    "candidate_count":len(records),
                    "anchored_interval":[round(lo,4),round(hi,4)]})
    return {"blocked_line_ids":sorted(blocked),
            "issues":issues}


def quarantine_conflicted_candidates(words, lines, candidates, *, enabled=True):
    """Flag structural conflicts while retaining every raw hypothesis."""
    if not enabled:
        return list(candidates),{"blocked_line_ids":[],"issues":[]}
    audit=assess_phrase_conflicts(words,lines,candidates)
    id_to_line={w["id"]:w["line_id"] for w in words}
    blocked=set(audit["blocked_line_ids"])
    revised=[]
    for candidate in candidates:
        c=dict(candidate)
        if id_to_line.get(c.get("word_id")) in blocked and c.get("status")!="rejected":
            c["status"]="rejected"
            c["rejection_reason"]="phrase_location_conflict"
        revised.append(c)
    return revised,audit
