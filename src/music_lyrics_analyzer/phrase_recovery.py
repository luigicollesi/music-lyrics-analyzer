"""Experimental phrase CTC: generate, audit, resolve; no in-loop mutation."""
from __future__ import annotations
from .phrase_windows import plan_phrase_windows
from .candidate_resolver import resolve_candidates, commit_resolved


def phrase_ctc_recovery(backend, audio16, words, lines, sections, *,
                        threshold=-2.5, score_window=2,
                        max_records=10000):
    """Analyze complete lyric lines using section-local acoustic evidence.

    Existing timestamps remain immutable while CTC candidates are generated.
    The resolver only commits additional missing words with a monotonic path.
    """
    all_candidates=[]
    attempts=[]
    for section in sections:
        bounds=section.get("window")
        if not bounds:
            continue
        sec_words=[word for word in words if word["section_id"]==section["id"]]
        sec_lines=[line for line in lines if line["section_id"]==section["id"]]
        windows,issues=plan_phrase_windows(
            sec_words,sec_lines,section_start=float(bounds[0]),
            section_end=float(bounds[1]))
        for issue in issues:
            attempts.append({"section_id":section["id"],**issue,
                             "status":"skipped"})
        for window in windows:
            selected=sec_words[window.first:window.last]
            entry={"section_id":section["id"],"window_id":window.id,
                   "start":window.start,"end":window.end,
                   "word_ids":[w["id"] for w in selected],
                   "unresolved_ids":list(window.unresolved_ids)}
            try:
                backend.align(audio16, selected, window.start,window.end,
                              threshold=threshold,score_window=score_window)
                diagnostic=getattr(backend,"last_diagnostic",{})
                records=diagnostic.get("candidate_records",[])
                entry["status"]=diagnostic.get("status","unknown")
                entry["rejections"]=diagnostic.get("rejected",{})
                entry["candidates"]=len(records)
                for record in records:
                    if len(all_candidates)>=max_records:
                        raise ValueError("CTC candidate record limit reached")
                    all_candidates.append({**record,
                        "window_id":window.id,
                        "section_id":section["id"]})
            except Exception as exc:
                entry["status"]="error"
                entry["error"]=str(exc)[:350]
            attempts.append(entry)
    solution=resolve_candidates(words,all_candidates)
    count=commit_resolved(words,solution)
    return {"recovered":count,
            "attempts":attempts,
            "candidates":all_candidates,
            "resolver":{"accepted_count":solution["accepted_count"],
                        "decisions":solution["decisions"]}}
