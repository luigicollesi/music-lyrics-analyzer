"""Experimental overlap search; no automatic word-time publication."""
from __future__ import annotations

from .overlap_windows import anchored_line_windows
from .phrase_conflicts import quarantine_conflicted_candidates
from .candidate_resolver import resolve_candidates
from .asr_line_search import propose_missing_line_locations


def overlap_ctc_review(backend, audio16, words, lines, *, seconds=8.,
                       hop=4., threshold=-2.5, max_records=10000,
                       asr_segments=None, asr_search=True):
    """Re-align only anchored whole lines on overlapping audio crops.

    This compares timing stability across crop borders. It does NOT discover
    completely missing lines (that requires independent ASR localization).
    """
    plans,skipped=anchored_line_windows(
        words,lines,start=0.,end=len(audio16)/16000,
        seconds=seconds,hop=hop)
    asr_plans=[]
    if asr_search and asr_segments:
        asr_plans,asr_issues=propose_missing_line_locations(
            words,lines,asr_segments,duration=len(audio16)/16000)
        skipped.extend(asr_issues)
    plans.extend(asr_plans)
    lookup={w["id"]:w for w in words}
    attempts=[{"status":"skipped",**issue} for issue in skipped]
    candidates=[]
    # Adjacent lyric lines can use the same audio crop. The MMS backend has
    # in-memory last-crop memoization to avoid repeating expensive inference.
    for plan in plans:
        selected=[lookup[wid] for wid in plan["word_ids"]]
        entry={"window_id":plan["window_id"],"line_id":plan["line_id"],
               "section_id":plan["section_id"],
               "start":plan["start"],"end":plan["end"],
               "anchor_count":plan["anchor_count"],
               "localization_source":plan.get("source","existing_word_anchors"),
               "asr_score":plan.get("asr_score")}
        try:
            backend.align_continuous(audio16,selected,plan["start"],plan["end"],
                                     threshold=threshold)
            diag=getattr(backend,"last_diagnostic",{})
            records=diag.get("candidate_records",[])
            entry["status"]=diag.get("status","unknown")
            entry["candidate_count"]=len(records)
            entry["blank_frame_ratio"]=diag.get("blank_frame_ratio")
            for record in records:
                if len(candidates)>=max_records:
                    raise ValueError("CTC candidate record limit reached")
                candidates.append({**record,"window_id":plan["window_id"],
                                   "line_id":plan["line_id"],
                                   "localization_source":entry["localization_source"],
                                   "asr_score":entry["asr_score"],
                                   "section_id":plan["section_id"]})
        except Exception as exc:
            entry["status"]="error"
            entry["error"]=str(exc)[:350]
        attempts.append(entry)
    candidates,conflicts=quarantine_conflicted_candidates(
        words,lines,candidates)
    resolution=resolve_candidates(words,candidates)
    # These are stability measurements of repeatedly aligned ANCHORED lines,
    # not proof of correctness, and never applied to the baseline timeline.
    per_word={}
    for c in candidates:
        if c.get("status")!="candidate":continue
        per_word.setdefault(c["word_id"],[]).append(c)
    stability=[]
    for wid,group in per_word.items():
        if len(group)<2:continue
        starts=[float(c["start"]) for c in group]
        ends=[float(c["end"]) for c in group]
        stability.append({
            "word_id":wid,"hypotheses":len(group),
            "start_range_seconds":round(max(starts)-min(starts),4),
            "end_range_seconds":round(max(ends)-min(ends),4)})
    return {
        "mode":"overlap","committed":False,"recovered":0,
        "attempts":attempts,"candidates":candidates,
        "phrase_conflicts":conflicts,
        "stability":stability,
        "asr_localization":{"proposed_windows":len(asr_plans),
                            "skipped":len(skipped)},
        "resolver":{"accepted_count":resolution["accepted_count"],
                    "decisions":resolution["decisions"],
                    "candidate_decisions":resolution["candidate_decisions"]}}
