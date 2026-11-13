"""Experimental phrase CTC: generate, audit, resolve; no in-loop mutation."""
from __future__ import annotations
from .phrase_windows import plan_phrase_windows
from .candidate_resolver import resolve_candidates, commit_resolved


def phrase_ctc_recovery(backend, audio16, words, lines, sections, *,
                        threshold=-2.5, score_window=2,
                        max_records=10000, mode="phrase", commit=True):
    """Analyze complete lyric lines using section-local acoustic evidence.

    Existing timestamps remain immutable while CTC candidates are generated.
    The resolver only commits additional missing words with a monotonic path.
    """
    all_candidates=[]
    attempts=[]
    known=[section for section in sections if section.get("window")]
    for section_idx, section in enumerate(known):
        bounds=section["window"]
        sec_words=[word for word in words if word["section_id"]==section["id"]]
        sec_lines=[line for line in lines if line["section_id"]==section["id"]]
        lo,hi=map(float,bounds)
        # Section windows inferred from *aligned* words often start after
        # an unresolved lyrical prefix. Expand cautiously, constrained by
        # adjacent sections, and expose the change in the attempt log.
        expanded=[]
        if sec_words and sec_words[0].get("start") is None:
            prev_end=(float(known[section_idx-1]["window"][1])
                      if section_idx else 0.)
            widened=max(prev_end,lo-2.5)
            if widened<lo:
                expanded.append({"edge":"start","from":lo,"to":widened})
                lo=widened
        if sec_words and sec_words[-1].get("start") is None:
            following=(float(known[section_idx+1]["window"][0])
                       if section_idx+1<len(known) else len(audio16)/16000)
            widened=min(following,hi+2.5)
            if widened>hi:
                expanded.append({"edge":"end","from":hi,"to":widened})
                hi=widened
        if expanded:
            attempts.append({"section_id":section["id"],
                             "status":"section_window_expanded",
                             "edges":expanded})
        windows,issues=plan_phrase_windows(
            sec_words,sec_lines,section_start=lo,
            section_end=hi)
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
                if mode == "continuous":
                    backend.align_continuous(
                        audio16, selected, window.start, window.end,
                        threshold=threshold,score_window=score_window)
                else:
                    backend.align(audio16, selected, window.start,window.end,
                                  threshold=threshold,score_window=score_window)
                diagnostic=getattr(backend,"last_diagnostic",{})
                records=diagnostic.get("candidate_records",[])
                entry["status"]=diagnostic.get("status","unknown")
                entry["rejections"]=diagnostic.get("rejected",{})
                entry["candidates"]=len(records)
                if mode=="continuous" and diagnostic.get("token_spans") is not None:
                    entry["token_spans"]=diagnostic["token_spans"]
                    entry["blank_frame_ratio"]=diagnostic.get("blank_frame_ratio")
                    entry["frame_duration"]=diagnostic.get("frame_duration")
                    entry["total_path_logprob"]=diagnostic.get("total_path_logprob")
                if diagnostic.get("continuous_reference") is not None:
                    ref=diagnostic["continuous_reference"]
                    entry["continuous_reference"]={
                        "status":ref["status"],
                        "token_count":len(ref["token_ids"]),
                        "diacritic_fallback":ref["diacritic_fallback"],
                        "capacity":ref["capacity"]}
                if diagnostic.get("phrase_tokenization_warning"):
                    entry["phrase_tokenization_warning"]=(
                        diagnostic["phrase_tokenization_warning"])
                if diagnostic.get("status")=="insufficient_frames":
                    entry["capacity"]={
                        "required_frames":diagnostic.get("required_frames"),
                        "available_frames":diagnostic.get("available_frames"),
                        "deficit_frames":diagnostic.get("deficit_frames")}
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
    count=commit_resolved(words,solution) if commit else 0
    return {"recovered":count,"mode":mode,"committed":commit,
            "attempts":attempts,
            "candidates":all_candidates,
            "resolver":{"accepted_count":solution["accepted_count"],
                        "decisions":solution["decisions"],
                        "candidate_decisions":solution["candidate_decisions"]}}
