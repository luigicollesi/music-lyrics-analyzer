"""Word-level vocal prosody from measured audio, never from lyric semantics.

F0 is estimated only in bounded, timestamp-anchored audio windows.
Pitch/energy are descriptive acoustics, not emotion, vocal identity or shouting.
"""
from __future__ import annotations

import math
import numpy as np


def _r(value, digits=3):
    return round(float(value), digits) if math.isfinite(float(value)) else None


def _contour(f0_hz, times, *, min_change_semitones=1.5):
    """Classify F0 trend using robust early/late medians, not instantaneous noise."""
    f0 = np.asarray(f0_hz, dtype=float)
    t = np.asarray(times, dtype=float)
    valid = np.isfinite(f0) & (f0 > 0) & np.isfinite(t)
    if np.count_nonzero(valid) < 4:
        return None, None
    f0, t = f0[valid], t[valid]
    k = max(2, len(f0) // 3)
    first, last = float(np.median(f0[:k])), float(np.median(f0[-k:]))
    delta = 12 * math.log2(last / first)
    # A note may bend or wobble, yet have the same beginning/end pitch:
    # report mixed for large excursions instead of calling it steady.
    spread = 12 * math.log2(float(np.percentile(f0, 90) / np.percentile(f0, 10)))
    if delta >= min_change_semitones:
        label = "rising"
    elif delta <= -min_change_semitones:
        label = "falling"
    elif spread >= 3.0:
        label = "variable"
    else:
        label = "steady"
    return label, _r(delta, 2)


def _rms_envelope(samples, sr, *, frame_seconds=.025, hop_seconds=.010):
    frame = max(64, round(sr * frame_seconds))
    hop = max(1, round(sr * hop_seconds))
    if len(samples) < frame:
        return np.array([math.sqrt(float(np.mean(samples.astype(float) ** 2)))]) if len(samples) else np.array([])
    values = [math.sqrt(float(np.mean(samples[i:i+frame].astype(float) ** 2)))
              for i in range(0, len(samples) - frame + 1, hop)]
    return np.asarray(values)


def _intensity(samples, sr):
    """Word RMS in dBFS, peak and relative energy trend, not human loudness."""
    if len(samples) == 0:
        return {"rms_dbfs": None, "peak_dbfs": None, "contour": None,
                "change_db": None, "attack_db": None}
    env = _rms_envelope(samples, sr)
    if len(env) == 0:
        return {"rms_dbfs": None, "peak_dbfs": None, "contour": None,
                "change_db": None, "attack_db": None}
    eps = 1e-8
    rms = math.sqrt(float(np.mean(samples.astype(float) ** 2)))
    peak = float(np.max(np.abs(samples)))
    k = max(1, len(env) // 3)
    early = float(np.median(env[:k]))
    late = float(np.median(env[-k:]))
    change = 20 * math.log10((late + eps) / (early + eps))
    if len(env) < 4 or max(early, late) < 1e-5:
        shape = None
    elif change > 3:
        shape = "crescendo"
    elif change < -3:
        shape = "decrescendo"
    else:
        shape = "steady"
    # Attack is a local change of measured energy, NOT "shouted" or emotion.
    attack = None
    if len(env) >= 2:
        levels = 20 * np.log10(env + eps)
        attack = max(0., float(np.max(np.diff(levels))))
    return {"rms_dbfs": _r(20 * math.log10(rms + eps), 2),
            "peak_dbfs": _r(20 * math.log10(peak + eps), 2),
            "contour": shape,
            "change_db": _r(change, 2),
            "attack_db": _r(attack, 2) if attack is not None else None}


def _groups(words, *, max_seconds=12., max_gap=.3):
    aligned = [w for w in words if w.get("start") is not None
               and w.get("end") is not None and w["end"] > w["start"]]
    aligned.sort(key=lambda w: w["start"])
    groups = []
    for word in aligned:
        if not groups or (word["end"] - groups[-1][-1]["start"] > max_seconds
                          or word["start"] - groups[-1][-1]["end"] > max_gap):
            groups.append([word])
        else:
            groups[-1].append(word)
    return groups


def _pitch_in_group(samples, sr, *, fmin, fmax, hop_length, frame_length):
    """One pYIN pass per bounded group (not one model run per word)."""
    import librosa
    f0, voiced, probability = librosa.pyin(
        samples, sr=sr, fmin=fmin, fmax=fmax,
        hop_length=hop_length, frame_length=frame_length,
        center=True, fill_na=np.nan)
    times = np.arange(len(f0), dtype=float) * hop_length / sr
    good = np.asarray(voiced, dtype=bool) & np.isfinite(f0) & (np.asarray(probability) >= .5)
    return times, np.asarray(f0, dtype=float), good, np.asarray(probability, dtype=float)


def analyze_vocal_prosody(audio, sr, words, *, source="mixture",
                          fmin=75., fmax=1000., hop_length=256,
                          frame_length=2048, pitch_extractor=None,
                          include_contour=True):
    """Attach measured prosody to each word that has accepted timing.

    The pitch extractor injection allows deterministic tests; production uses
    librosa.pyin. A low voiced fraction never becomes a guessed pitch.
    """
    y = np.asarray(audio, dtype=np.float32)
    if y.ndim != 1 or sr <= 0 or y.size == 0:
        raise ValueError("Prosody requires nonempty mono PCM audio.")
    if not (0 < fmin < fmax <= sr / 2):
        raise ValueError("Invalid pitch limits for audio sample rate.")
    if frame_length <= sr / fmin:
        raise ValueError("Pitch frame too short for selected fmin.")
    if hop_length <= 0 or hop_length > frame_length:
        raise ValueError("Invalid pitch hop length.")
    duration = y.size / sr
    metrics = {"analyzed_words":0, "pitch_resolved_words":0,
               "unresolved_words":0, "source":source,
               "pitch_method":"pyin", "hop_seconds":_r(hop_length / sr, 6)}
    warnings=[]
    for word in words:
        word["vocal"] = None
        start,end=word.get("start"),word.get("end")
        if start is None or end is None or not (
            math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= duration + .005):
            metrics["unresolved_words"] += 1
            continue
        x = y[int(start*sr):min(len(y),int(end*sr))]
        loudness = _intensity(x,sr)
        word["vocal"] = {
            "source":source,
            "reliability":"low" if source=="mixture" else "estimated",
            "pitch":{"mean_hz":None,"min_hz":None,"max_hz":None,
                     "voiced_fraction":None,"voicing_probability":None,
                     "contour":None,"change_semitones":None,
                     "samples":[] if include_contour else None},
            "intensity":loudness,
            "delivery":{"sustained":None,"breathy":None,"shouted":None,"vibrato":None},
        }
        metrics["analyzed_words"] += 1
    extractor = pitch_extractor or _pitch_in_group
    for group in _groups(words):
        start=max(0., group[0]["start"] - .10)
        end=min(duration,group[-1]["end"] + .10)
        chunk=y[int(start*sr):min(len(y),int(end*sr))]
        if len(chunk) < frame_length:
            warnings.append("Pitch inconclusivo: janela curta em " + group[0]["id"])
            continue
        try:
            times,f0,voiced,prob=extractor(chunk,sr,fmin=fmin,fmax=fmax,
                 hop_length=hop_length,frame_length=frame_length)
            times=np.asarray(times)+start
            f0=np.asarray(f0,dtype=float)
            voiced=np.asarray(voiced,dtype=bool)
            prob=np.asarray(prob,dtype=float)
            if not (len(times)==len(f0)==len(voiced)==len(prob)):
                raise ValueError("Pitch extractor returned incompatible frame arrays.")
        except Exception as exc:
            warnings.append(f"Pitch indisponível no grupo {group[0]['id']}: {exc}")
            continue
        for word in group:
            p=word["vocal"]["pitch"]
            indices=np.flatnonzero((times>=word["start"]) & (times<word["end"]))
            if indices.size == 0:
                continue
            good=indices[voiced[indices] & np.isfinite(f0[indices]) & (f0[indices]>0)]
            p["voiced_fraction"]=_r(len(good)/len(indices),3)
            p["voicing_probability"]=_r(np.mean(prob[indices]),3)
            if len(good)<3 or len(good)/len(indices)<.4:
                continue
            pitches=f0[good]
            p["mean_hz"]=_r(np.median(pitches),2)
            p["min_hz"]=_r(np.percentile(pitches,10),2)
            p["max_hz"]=_r(np.percentile(pitches,90),2)
            p["contour"],p["change_semitones"]=_contour(pitches,times[good])
            if include_contour:
                p["samples"]=[{"time":_r(t,3),"hz":_r(h,2)}
                              for t,h in zip(times[good],pitches)]
            metrics["pitch_resolved_words"] += 1
    if source=="mixture":
        warnings.append("Pitch em áudio mixado pode acompanhar instrumentos; recomenda-se --vocals-path ou --separator demucs.")
    return metrics,warnings
