"""Overlapping audio-window planning with anchored lyric evidence.

Never feed an arbitrary lyric phrase into every audio crop. Only select
occurrence-specific lines whose established timestamps intersect that crop.
Unanchored lines are reported for a future independent ASR search.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class SlidingWindow:
    id: str
    start: float
    end: float


def sliding_windows(start: float, end: float, *, seconds=8., hop=4.):
    if not all(isfinite(float(v)) for v in (start,end,seconds,hop)):
        raise ValueError("Sliding window parameters must be finite.")
    if not 0 <= start < end or not 0 < hop <= seconds <= 20:
        raise ValueError("Need 0<hop<=window<=20 and 0<=start<end.")
    result=[]
    pos=float(start)
    index=0
    while pos < end - 1e-7:
        stop=min(end,pos+seconds)
        result.append(SlidingWindow(f"overlap-{index:04d}",round(pos,4),round(stop,4)))
        if stop>=end-1e-7:
            break
        pos+=hop
        index+=1
    return result


def anchored_line_windows(words, lines, *, start, end, seconds=8., hop=4.,
                          edge_margin=.25, minimum_anchor_words=2):
    """Select whole anchored lines for crops with enough context.

    This is *not* independent phrase discovery. A candidate phrase must
    already have >=2 timestamped word anchors in the matching lyric line.
    Timed span must lie inside window's central safe region. Each line is
    decoded once per viable crop, enabling border-stability comparisons.
    """
    if not 0 <= edge_margin < seconds/2:
        raise ValueError("Invalid edge margin.")
    windows=sliding_windows(start,end,seconds=seconds,hop=hop)
    by_id={w["id"]:w for w in words}
    output=[]
    skipped=[]
    for line in lines:
        line_words=[by_id[k] for k in line["word_ids"] if k in by_id]
        if not line_words:continue
        timed=[w for w in line_words if w.get("start") is not None
               and w.get("end") is not None]
        if len(timed)<minimum_anchor_words:
            skipped.append({"line_id":line["id"],"reason":"insufficient_independent_anchors",
                            "anchor_count":len(timed)})
            continue
        first=min(float(w["start"]) for w in timed)
        last=max(float(w["end"]) for w in timed)
        if not 0 <= first < last:
            skipped.append({"line_id":line["id"],"reason":"invalid_anchors"})
            continue
        matched=0
        for window in windows:
            if first < window.start+edge_margin or last > window.end-edge_margin:
                continue
            output.append({
                "window_id":window.id,
                "line_id":line["id"],
                "section_id":line["section_id"],
                "start":window.start,"end":window.end,
                "word_ids":[w["id"] for w in line_words],
                "anchor_count":len(timed)})
            matched+=1
        if matched==0:
            skipped.append({"line_id":line["id"],"reason":"no_containing_window",
                            "anchor_count":len(timed)})
    return output,skipped
