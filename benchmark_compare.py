"""Compare two lyric-analysis JSON outputs without conflating coverage and accuracy."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def compare(a, b):
    base={w["id"]:w for w in a["words"]}
    newer={w["id"]:w for w in b["words"]}
    if set(base) != set(newer):
        raise ValueError("Different word IDs: outputs are not directly comparable")
    if any(base[key]["text"] != newer[key]["text"] for key in base):
        raise ValueError("Different lyrics: text changed between runs")
    old={key for key,word in base.items() if word.get("start") is not None}
    new={key for key,word in newer.items() if word.get("start") is not None}
    tentative={key for key in new
               if newer[key].get("alignment",{}).get("status")=="tentative"}
    return {
        "total":len(base),
        "baseline_timed":len(old),
        "candidate_timed":len(new),
        "newly_timed":len(new-old),
        "lost_timing":len(old-new),
        "tentative":len(tentative),
        "coverage_before":round(len(old)/max(1,len(base)),4),
        "coverage_after":round(len(new)/max(1,len(base)),4),
        "newly_timed_word_ids":sorted(new-old),
        "lost_word_ids":sorted(old-new),
        "warning":"Coverage does not establish word timestamp accuracy."
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline",type=Path)
    parser.add_argument("candidate",type=Path)
    args=parser.parse_args()
    result=compare(json.loads(args.baseline.read_text(encoding="utf-8")),
                   json.loads(args.candidate.read_text(encoding="utf-8")))
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
