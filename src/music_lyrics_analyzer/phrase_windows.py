"""Phrase-level acoustic windows. Never infer locations from BPM."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class AlignmentWindow:
    id: str
    start: float
    end: float
    first: int
    last: int
    unresolved_ids: tuple[str, ...]
    boundary_source: str

    def as_dict(self):
        return vars(self)


def plan_phrase_windows(words, lines, *, section_start, section_end,
                        min_seconds=2., max_seconds=12., padding=.35):
    """Plan overlapping phrase crops from existing aligned line-local evidence.

    The first/last timed words are *soft* context only. The supplied section
    window is the absolute hard boundary. When a line begins before the first
    timed word, a wider crop is allowed, without inventing an earlier start.
    """
    if not 0 <= section_start < section_end:
        return [], [{"reason":"invalid_section_window"}]
    line_index={l["id"]:i for i,l in enumerate(lines)}
    by_line={}
    for index, word in enumerate(words):
        by_line.setdefault(word["line_id"], []).append(index)
    windows, issues=[],[]
    for line in lines:
        indices=by_line.get(line["id"], [])
        missing=[i for i in indices if words[i].get("start") is None]
        if not missing: continue
        if not indices: continue
        # Expand across adjacent lines if necessary, but only within section.
        first,last=indices[0], indices[-1]+1
        chosen=[words[i] for i in range(first,last)]
        timed=[w for w in chosen if w.get("start") is not None]
        if timed:
            left=min(float(w["start"]) for w in timed)
            right=max(float(w["end"]) for w in timed)
            lo=max(section_start,left-padding)
            hi=min(section_end,right+padding)
            if hi-lo<min_seconds:
                center=(hi+lo)/2
                lo=max(section_start,center-min_seconds/2)
                hi=min(section_end,center+min_seconds/2)
        else:
            # Without an acoustic anchor for this line, neighboring lines
            # provide conservative boundaries; never interpolate lyric time.
            prev=[w for w in words[:first] if w.get("end") is not None]
            nxt=[w for w in words[last:] if w.get("start") is not None]
            lo=max(section_start,float(prev[-1]["end"]) if prev else section_start)
            hi=min(section_end,float(nxt[0]["start"]) if nxt else section_end)
        if hi<=lo or hi-lo>max_seconds:
            issues.append({"line_id":line["id"],"reason":"no_safe_phrase_window",
                           "window":[round(lo,3),round(hi,3)]})
            continue
        # Include words from directly adjacent aligned lines only if their
        # measured timestamps actually overlap this crop.
        context_first=first
        while context_first>0 and words[context_first-1]["section_id"]==line["section_id"]:
            other=words[context_first-1]
            if other.get("end") is None or float(other["end"])<lo: break
            context_first-=1
        context_last=last
        while context_last<len(words) and words[context_last]["section_id"]==line["section_id"]:
            other=words[context_last]
            if other.get("start") is None or float(other["start"])>hi: break
            context_last+=1
        if context_last-context_first>50:
            issues.append({"line_id":line["id"],"reason":"too_many_tokens"})
            continue
        windows.append(AlignmentWindow(
            id=f"phrase-{line['id']}",start=round(lo,3),end=round(hi,3),
            first=context_first,last=context_last,
            unresolved_ids=tuple(words[i]["id"] for i in missing),
            boundary_source="line_anchors" if timed else "neighbor_lines"))
    return windows,issues
