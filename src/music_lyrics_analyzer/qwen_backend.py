"""Optional Qwen3 ASR auxiliary transcript with explicit GPU capability checks.

Qwen ASR is a phrase *localizer*, not a word-timestamp authority.
No model downloads or GPU initialization occur on module import.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
import numpy as np

MODELS={"0.6b":"Qwen/Qwen3-ASR-0.6B",
        "1.7b":"Qwen/Qwen3-ASR-1.7B"}


@dataclass(frozen=True)
class DevicePlan:
    device: str
    dtype: str
    name: str
    capability: tuple[int,int] | None
    total_vram_gb: float | None


def choose_device(torch, requested="cuda", *, minimum_vram_gb=0.):
    """Fail closed on unsupported Pascal wheels, insufficient VRAM, or CPU mismatch."""
    if requested not in ("cuda","cpu","auto"):
        raise ValueError("Device must be cpu, cuda or auto.")
    if requested=="cpu":
        return DevicePlan("cpu","float32","CPU",None,None)
    if not torch.cuda.is_available():
        if requested=="cuda":
            raise RuntimeError("CUDA indisponível. Verifique driver, torch e GPU. "
                               "Use --qwen-device cpu ou instale um build CUDA compatível.")
        return DevicePlan("cpu","float32","CPU",None,None)
    major,minor=tuple(torch.cuda.get_device_capability(0))
    name=str(torch.cuda.get_device_name(0))
    mem=round(torch.cuda.get_device_properties(0).total_memory/1024**3,2)
    supported_arches=(torch.cuda.get_arch_list()
                       if hasattr(torch.cuda,"get_arch_list") else [])
    arch=f"sm_{major}{minor}"
    # NVIDIA cubins are forward-compatible within the same major
    # compute capability: sm_60 can execute on Pascal sm_61.
    # A higher minor target (sm_61 on sm_60) is NOT compatible.
    built=[]
    for entry in supported_arches:
        if not entry.startswith("sm_"):
            continue
        digits=entry[3:]
        if digits.isdigit() and len(digits)>=2:
            built_major=int(digits[:-1])
            built_minor=int(digits[-1])
            built.append((built_major,built_minor))
    compatible=any(m==major and n<=minor for m,n in built)
    if supported_arches and not compatible:
        msg=(f"GPU {name} ({arch}) não suportada pelos kernels do PyTorch "
             f"instalado ({', '.join(supported_arches)}). "
             "Instale uma versão com suporte a essa arquitetura ou use CPU.")
        if requested=="cuda":
            raise RuntimeError(msg)
        return DevicePlan("cpu","float32","CPU",None,None)
    if mem < minimum_vram_gb:
        if requested=="cuda":
            raise RuntimeError(
                f"VRAM insuficiente para o orçamento configurado: "
                f"{mem:.2f} GiB disponíveis; requisito {minimum_vram_gb:.2f} GiB. "
                "Reduza o modelo ou execute na CPU.")
        return DevicePlan("cpu","float32","CPU",None,None)
    return DevicePlan("cuda:0","float16",name,(major,minor),mem)


def transcribe_qwen_windows(audio16, *, model_size="0.6b",
                            device="cuda", chunk_seconds=16.,
                            hop_seconds=None, language="Portuguese",
                            max_new_tokens=256, minimum_vram_gb=0.,
                            progress=None):
    """Generate approximate chunk-level ASR text intervals.

    Crucially, crop start/end are the *entire chunk* rather than imagined
    word/phrase-level timestamps. These coarse segments must not become
    definitive evidence of exact word boundaries.
    """
    if model_size not in MODELS:
        raise ValueError("Unknown Qwen model size.")
    if not 1<=chunk_seconds<=30:
        raise ValueError("chunk_seconds must be between 1 and 30.")
    if hop_seconds is None:hop_seconds=chunk_seconds/2
    if not 0<hop_seconds<=chunk_seconds:
        raise ValueError("Invalid ASR hop.")
    try:
        import torch
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise RuntimeError(
            "Qwen3-ASR não instalado. Execute: pip install -e '.[qwen]'") from exc
    plan=choose_device(torch,device,minimum_vram_gb=minimum_vram_gb)
    dtype=torch.float16 if plan.device.startswith("cuda") else torch.float32
    model=Qwen3ASRModel.from_pretrained(
        MODELS[model_size],dtype=dtype,device_map=plan.device,
        max_inference_batch_size=1,max_new_tokens=max_new_tokens)
    sr=16000
    sound=np.asarray(audio16,dtype=np.float32)
    from .overlap_windows import sliding_windows
    crops=sliding_windows(0.,len(sound)/sr,seconds=chunk_seconds,
                          hop=hop_seconds)
    output=[]
    for window in crops:
        a=max(0,int(round(window.start*sr)))
        b=min(len(sound),int(round(window.end*sr)))
        if b-a<sr:continue
        results=model.transcribe(audio=(sound[a:b],sr),language=language)
        if not results:continue
        phrase=str(results[0].text).strip()
        if not phrase:continue
        output.append({"start":window.start,"end":window.end,
                       "text":phrase,"source":"qwen3_asr_chunk",
                       "window_id":window.id,
                       "time_precision":"chunk_only"})
        if progress:progress(window,phrase)
    return output,plan


def transcribe_qwen_full(audio16, *, model_size="0.6b",
                         device="cuda", language="Portuguese",
                         max_new_tokens=2048, minimum_vram_gb=0.,
                         max_audio_seconds=600.):
    """One Qwen transcription request for the whole audio, without cropping.

    The result contains text, NOT timestamped phrases. A generation output
    near the requested token cap is flagged for review rather than trusted.
    OOM errors are propagated; the caller may explicitly retry windowed mode.
    """
    if model_size not in MODELS:
        raise ValueError("Unknown Qwen model size.")
    if not 128 <= max_new_tokens <= 8192:
        raise ValueError("max_new_tokens must be between 128 and 8192.")
    samples=np.asarray(audio16,dtype=np.float32)
    if samples.ndim != 1 or samples.size < 16000:
        raise ValueError("Expected mono audio with at least 1 second at 16kHz.")
    duration=len(samples)/16000.
    if duration > max_audio_seconds:
        raise ValueError("Full ASR audio exceeds the configured safety duration.")
    try:
        import torch
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        raise RuntimeError("Install optional Qwen backend: pip install -e '.[qwen]'") from exc
    plan=choose_device(torch,device,minimum_vram_gb=minimum_vram_gb)
    dtype=torch.float16 if plan.device.startswith("cuda") else torch.float32
    model=Qwen3ASRModel.from_pretrained(
        MODELS[model_size],dtype=dtype,device_map=plan.device,
        max_inference_batch_size=1,max_new_tokens=max_new_tokens)
    try:
        result=model.transcribe(audio=(samples,16000),language=language)
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            "Qwen full-audio inference exceeded CUDA VRAM. "
            "Retry with --qwen-mode windowed or --qwen-device cpu."
        ) from exc
    if not result or not str(result[0].text).strip():
        raise RuntimeError("Qwen full-audio returned no transcription.")
    text=str(result[0].text).strip()
    return {
        "mode":"full","text":text,"duration_seconds":round(duration,3),
        "model":MODELS[model_size],"device":plan.device,
        "max_new_tokens":max_new_tokens,
        "word_count":len(text.split()),
        "token_limit_risk":len(text.split()) >= max_new_tokens//3,
        "time_precision":"none",
        "status":"transcribed_unverified",
    },plan
