"""Resolve independent acoustic candidate hypotheses without cascade effects."""
from __future__ import annotations
from collections import defaultdict
from math import isfinite


def resolve_candidates(words, candidates, *, maximum_overlap=.04,
                       allow_low_score=False):
    """Compute proposals on a snapshot; commit only once after optimization.

    Existing alignments are soft for *candidate generation* but hard for this
    conservative commit pass. The source words are never mutated here.
    Candidates are dicts with word_id/start/end/score/status. Untrusted
    candidates are recorded and may remain unaccepted.
    """
    groups=defaultdict(list)
    for candidate in candidates:
        if candidate.get("start") is None or candidate.get("end") is None:
            continue
        a,b=float(candidate["start"]),float(candidate["end"])
        if not (isfinite(a) and isfinite(b) and 0<=a<b):
            continue
        if candidate.get("status")=="rejected" and not allow_low_score:
            continue
        groups[candidate["word_id"]].append(candidate)
    accepted={}
    decisions=[]
    # Process *runs* of missing words. Never let a chosen tentative word
    # become a constraint on a future run.
    i=0
    while i<len(words):
        if words[i].get("start") is not None:
            i+=1
            continue
        j=i
        while j<len(words) and words[j].get("start") is None:
            j+=1
        left=words[i-1]["end"] if i and words[i-1].get("end") is not None else None
        right=words[j]["start"] if j<len(words) and words[j].get("start") is not None else None
        # DP includes explicit skip states, so an unsupported word remains
        # unresolved without forcing later timestamps into its slot.
        states=[(0.,float(left) if left is not None else -1.,{},[])]
        for word in words[i:j]:
            alternatives=groups.get(word["id"],[])
            future=[]
            for quality,prev_end,chosen,history in states:
                future.append((quality,prev_end,chosen,history+[(word["id"],"unresolved")]))
                for candidate in alternatives:
                    a,b=float(candidate["start"]),float(candidate["end"])
                    if a < prev_end-maximum_overlap:
                        continue
                    if right is not None and b > float(right)+maximum_overlap:
                        continue
                    # Prefer short, plausible, acoustically supported crops.
                    score=float(candidate.get("score", -5))
                    utility=2.+max(-6.,min(0.,score))*.20
                    proposed=dict(chosen)
                    proposed[word["id"]]=candidate
                    future.append((quality+utility,b,proposed,
                                   history+[(word["id"],"accepted")]))
            # Keep best state per end time; cap to prevent combinatorial blowup.
            future.sort(key=lambda state:state[0],reverse=True)
            states=future[:96]
        if states:
            best=max(states,key=lambda state:state[0])
            accepted.update(best[2])
            decisions.extend({"word_id":wid,"status":status}
                             for wid,status in best[3])
        i=j
    for word in words:
        if word.get("start") is None and word["id"] not in accepted and not any(
                record["word_id"]==word["id"] for record in decisions):
            decisions.append({"word_id":word["id"],"status":"unresolved"})
    # Audit the *second* stage independently from decoder rejections.
    position={word["id"]:i for i,word in enumerate(words)}
    candidate_decisions=[]
    for candidate in candidates:
        word_id=candidate.get("word_id")
        idx=position.get(word_id)
        if idx is None:
            reason="unknown_word"
        elif candidate.get("status")=="rejected":
            reason="decoder_rejected"
        elif word_id in accepted and candidate is accepted[word_id]:
            reason="accepted"
        elif words[idx].get("start") is not None:
            reason="already_aligned"
        elif candidate.get("start") is None or candidate.get("end") is None:
            reason="invalid_interval"
        else:
            left=next((w["end"] for w in reversed(words[:idx])
                       if w.get("end") is not None),None)
            right=next((w["start"] for w in words[idx+1:]
                        if w.get("start") is not None),None)
            if ((left is not None and float(candidate["start"])<float(left)-maximum_overlap)
                or (right is not None and float(candidate["end"])>float(right)+maximum_overlap)):
                reason="anchor_conflict"
            else:
                reason="not_selected"
        candidate_decisions.append({
            "word_id":word_id,
            "window_id":candidate.get("window_id"),
            "resolution":reason})
    return {"accepted":accepted,"decisions":decisions,
            "candidate_decisions":candidate_decisions,
            "accepted_count":len(accepted)}


def commit_resolved(words, solution):
    """One-time application of the resolver result. Existing values win."""
    updated=0
    by_id={w["id"]:w for w in words}
    for word_id,candidate in solution["accepted"].items():
        word=by_id[word_id]
        if word["start"] is not None:continue
        a,b=float(candidate["start"]),float(candidate["end"])
        word["start"],word["end"]=round(a,3),round(b,3)
        word["duration"]=round(b-a,3)
        word["alignment"]={"status":"tentative","confidence":None,
                           "method":"ctc-phrase",
                           "acoustic_score":candidate.get("score")}
        updated+=1
    return updated


def timeline_issues(words, *, duration):
    """Do not silently mutate existing timestamps to hide regressions."""
    problems=[]
    previous=None
    for word in words:
        a,b=word.get("start"),word.get("end")
        if a is None:continue
        if not 0<=float(a)<float(b)<=duration+.05:
            problems.append({"word_id":word["id"],"reason":"invalid_bounds"})
        if previous is not None and float(a)<previous["start"]-.04:
            problems.append({"word_id":word["id"],
                             "previous_id":previous["id"],
                             "reason":"start_regression"})
        previous={"id":word["id"],"start":float(a)}
    return problems
