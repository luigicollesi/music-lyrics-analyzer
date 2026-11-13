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


class _QwenOffloadDeviceProxy:
    """Expose the real CUDA execution device to the Qwen wrapper.

    Accelerate cpu_offload intentionally replaces model parameters with meta
    placeholders. Qwen's wrapper reads self.model.device to move token/audio
    inputs, which otherwise sends tensors to meta and fails during generation.
    The proxy only corrects that public-facing device, preserving the
    underlying model's Accelerate hooks and its generate implementation.
    """
    def __init__(self, module, execution_device, dtype):
        object.__setattr__(self,"_wrapped",module)
        object.__setattr__(self,"device",execution_device)
        object.__setattr__(self,"dtype",dtype)

    def __getattr__(self, name):
        return getattr(self._wrapped,name)


def transcribe_qwen_full(audio16, *, model_size="0.6b",
                         device="cuda", language="Portuguese",
                         max_new_tokens=2048, minimum_vram_gb=0.,
                         max_audio_seconds=600., gpu_memory_gb=2.5,
                         cpu_memory_gb=12., memory_report=False,
                         encoder_memory_trace=False,
                         offload_granularity="tensor", transfer_report=False,
                         resident_gpu_weights_gib=1.0,
                         activation_reserve_gib=1.5, placement_audit=False,
                         placement_strategy="auto"):
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
    from .inference_runtime.memory_profiler import MemoryProfiler
    profiler=MemoryProfiler(torch,enabled=memory_report)
    profiler.mark("before_model_load")
    if placement_strategy not in ("auto","decoder"):
        raise ValueError("Unknown Qwen placement strategy.")
    if placement_strategy=="decoder" and device!="resident":
        raise ValueError("Decoder-first placement requires --qwen-device resident.")
    decoder_plan=None
    if device=="resident":
        from .inference_runtime.residency_manager import budget_for_residency
        plan=choose_device(torch,"cuda",minimum_vram_gb=minimum_vram_gb)
        placement=budget_for_residency(
            physical_vram_gib=plan.total_vram_gb,
            weights_gib=resident_gpu_weights_gib,
            reserve_gib=activation_reserve_gib,
            cpu_memory_gib=cpu_memory_gb)
        if placement_strategy=="decoder":
            placement={"device_map":"cpu","low_cpu_mem_usage":True}
        dtype=torch.float16
    elif device=="streamed":
        # Genuine full-model per-forward parameter streaming: weights live
        # in RAM and Accelerate moves individual module weights to CUDA
        # when executed, then removes them after each forward.
        from accelerate import cpu_offload
        plan=choose_device(torch,"auto",minimum_vram_gb=minimum_vram_gb)
        # If CUDA is unavailable, the same full-song request runs on CPU,
        # without attaching GPU hooks. CPU float32 avoids unsupported FP16 ops.
        dtype=torch.float16 if plan.device.startswith("cuda") else torch.float32
        placement=dict(device_map="cpu",low_cpu_mem_usage=True)
    elif device=="hybrid":
        if not 0.5<=gpu_memory_gb<=64 or not 1<=cpu_memory_gb<=1024:
            raise ValueError("Invalid hybrid GPU or CPU memory budget.")
        try:
            import accelerate  # noqa: F401 -- required by HF device_map
        except ImportError as exc:
            raise RuntimeError("Hybrid mode requires accelerate: pip install -e '.[qwen]'") from exc
        plan=choose_device(torch,"cuda",minimum_vram_gb=minimum_vram_gb)
        if gpu_memory_gb>=plan.total_vram_gb:
            raise ValueError("Hybrid GPU budget must be below physical VRAM.")
        placement=dict(device_map="auto",max_memory={
            0:f"{int(gpu_memory_gb*1024)}MiB",
            "cpu":f"{int(cpu_memory_gb*1024)}MiB"},
            low_cpu_mem_usage=True)
        dtype=torch.float16
    else:
        plan=choose_device(torch,device,minimum_vram_gb=minimum_vram_gb)
        dtype=torch.float16 if plan.device.startswith("cuda") else torch.float32
        placement=dict(device_map=plan.device)
    model=Qwen3ASRModel.from_pretrained(
        MODELS[model_size],dtype=dtype,**placement,
        max_inference_batch_size=1,max_new_tokens=max_new_tokens)
    profiler.mark("after_model_load")
    if device=="resident" and placement_strategy=="decoder":
        from .inference_runtime.decoder_placement import apply_decoder_priority
        model.model,decoder_plan=apply_decoder_priority(
            model.model,gpu_weights_gib=resident_gpu_weights_gib,
            reserve_gib=activation_reserve_gib,
            physical_vram_gib=plan.total_vram_gb)
        model.device=torch.device("cuda:0")
        print("[Qwen decoder priority] "+str(decoder_plan),
              flush=True)
    if device=="streamed" and plan.device.startswith("cuda"):
        execution_device=torch.device("cuda:0")
        from .inference_runtime.weight_manager import attach_weight_offload
        from .inference_runtime.qwen_adapter import validate_model_for_generation
        attach_weight_offload(model.model,execution_device,
                              granularity=offload_granularity)
        offload_status=validate_model_for_generation(model.model)
        # The upstream Qwen wrapper uses self.model.device to place input
        # tensors; after cpu_offload that value becomes 'meta'. Expose the
        # real execution device without changing the underlying HF model.
        model.model=_QwenOffloadDeviceProxy(
            model.model,execution_device,torch.float16)
        model.device=execution_device
    profiler.mark("after_offload_setup")
    device_map=getattr(model.model,"hf_device_map",None)
    from .inference_runtime.residency_manager import residency_summary
    placement_summary=residency_summary(device_map)
    if placement_audit:
        import json
        from .inference_runtime.placement_audit import audit_placement
        placement_details=audit_placement(model.model)
        print("[Qwen placement] "+
              json.dumps(placement_details,ensure_ascii=False),flush=True)
    else:
        placement_details=None
    if plan.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(0)
    from contextlib import nullcontext
    if transfer_report:
        from .inference_runtime.transfer_profiler import TransferProfiler
        transfer_profiler=TransferProfiler()
        transfer_context=transfer_profiler.observe()
    else:
        transfer_profiler=None
        transfer_context=nullcontext()
    if encoder_memory_trace:
        from .inference_runtime.encoder_profiler import EncoderMemoryTrace
        encoder_trace=EncoderMemoryTrace(torch)
        tracing=encoder_trace.observe(model.model)
    else:
        encoder_trace=None
        tracing=nullcontext()
    try:
        with tracing, transfer_context:
            result=model.transcribe(audio=(samples,16000),language=language)
    except torch.cuda.OutOfMemoryError as exc:
        profiler.mark("oom_during_transcribe")
        import json
        details=(" Memory checkpoints: "+
                 json.dumps(profiler.to_dict(),ensure_ascii=False)) if memory_report else ""
        if transfer_profiler is not None:
            details+=" Transfer report: "+json.dumps(transfer_profiler.report())
        if encoder_trace is not None:
            details+=" Encoder trace: "+json.dumps(
                encoder_trace.to_dict(),ensure_ascii=False)
        raise RuntimeError(
            "Qwen full-audio inference exceeded CUDA VRAM. "
            "Weight streaming does not eliminate audio-encoder activations. "
            "Retry --qwen-mode windowed or --qwen-device cpu." + details
        ) from exc
    profiler.mark("after_transcribe")
    if not result or not str(result[0].text).strip():
        raise RuntimeError("Qwen full-audio returned no transcription.")
    text=str(result[0].text).strip()
    return {
        "mode":"full","text":text,"duration_seconds":round(duration,3),
        "model":MODELS[model_size],"device":device if device in ("hybrid","streamed","resident") else plan.device,
        "gpu_memory_budget_gb":(resident_gpu_weights_gib if device=="resident" else gpu_memory_gb if device=="hybrid" else None),
        "activation_reserve_gb":activation_reserve_gib if device=="resident" else None,
        "residency_summary":placement_summary if device=="resident" else None,
        "placement_audit":placement_details,
        "decoder_placement":decoder_plan,
        "cpu_memory_budget_gb":cpu_memory_gb if device=="hybrid" else None,
        "device_map":device_map,
        "offload_granularity":offload_granularity if device=="streamed" else None,
        "weight_offload": ("accelerate_cpu_offload_per_forward"
                           if device=="streamed" and plan.device.startswith("cuda")
                           else "cpu_only" if device=="streamed"
                           else "static_device_map"),
        "peak_gpu_allocated_gb":round(torch.cuda.max_memory_allocated(0)/1024**3,3)
            if plan.device.startswith("cuda") else None,
        "max_new_tokens":max_new_tokens,
        "word_count":len(text.split()),
        "token_limit_risk":len(text.split()) >= max_new_tokens//3,
        "memory_report":profiler.to_dict() if memory_report else None,
        "transfer_report":transfer_profiler.report() if transfer_profiler else None,
        "encoder_memory_trace":encoder_trace.to_dict() if encoder_trace is not None else None,
        "offload_integrity":offload_status if device=="streamed" and plan.device.startswith("cuda") else None,
        "time_precision":"none",
        "status":"transcribed_unverified",
    },plan
