"""Local, explicit speech/canto smoke test of MMS language adapter.

This is NOT a CI test: it downloads MMS weights and requires actual reference
audio. It does not modify the song, timestamps, or user files.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from music_lyrics_analyzer.cli import read_audio
from music_lyrics_analyzer.ctc_backend import MMSBackend


def normalize(text):
    from unicodedata import normalize as unicode_normalize, category
    value=unicode_normalize("NFKD",text.casefold())
    return "".join(c for c in value if category(c)!="Mn" and
                   (c.isalpha() or c.isspace())).split()


def edit_distance(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        row=[i]
        for j,y in enumerate(b,1):
            row.append(min(prev[j]+1,row[-1]+1,prev[j-1]+(x!=y)))
        prev=row
    return prev[-1]


def main():
    parser=argparse.ArgumentParser(description="Test MMS Portuguese free decoding.")
    parser.add_argument("--audio",required=True,help="Known spoken/sung Portuguese sample.")
    parser.add_argument("--expected",required=True,help="Human-verified phrase in audio.")
    parser.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    parser.add_argument("--model",default="facebook/mms-1b-all")
    parser.add_argument("--language",default="por")
    args=parser.parse_args()
    path=Path(args.audio)
    if not path.is_file():
        parser.error("Sample audio does not exist.")
    samples,sr=read_audio(path,16000)
    backend=MMSBackend(model_id=args.model,language=args.language,
                       device=args.device)
    emission=backend.infer(samples)
    ids=np.argmax(emission,axis=-1)
    predicted=backend.processor.tokenizer.decode(ids.tolist(),
                                                  skip_special_tokens=True)
    reference_words=normalize(args.expected)
    predicted_words=normalize(predicted)
    output={
        "model_integrity":backend.integrity.to_dict(),
        "audio_duration":round(len(samples)/sr,4),
        "emission_frames":len(emission),
        "expected":args.expected,
        "predicted":predicted,
        "reference_words":len(reference_words),
        "word_error_rate":round(
            edit_distance(reference_words,predicted_words)/
            max(1,len(reference_words)),4),
        "status":"requires_human_review",
        "note":"WER is sample-specific; free decoding does not validate word boundaries."
    }
    print(json.dumps(output,ensure_ascii=False,indent=2))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
