"""Weight placement policies for Qwen inference.

Layer preloading reduces Python-side per-tensor hook calls but increases
instantaneous GPU memory usage. It does not retain layers between tokens.
"""
from __future__ import annotations

SUPPORTED_GRANULARITIES=("tensor","layer")


def offload_options(granularity):
    if granularity not in SUPPORTED_GRANULARITIES:
        raise ValueError(f"Unknown Qwen offload granularity: {granularity}")
    if granularity=="tensor":
        return {}
    # Only the repeated decoder block, not the entire audio encoder.
    return {"preload_module_classes":["Qwen3ASRThinkerTextDecoderLayer"]}


def attach_weight_offload(model, execution_device, *, granularity="tensor"):
    from accelerate import cpu_offload
    options=offload_options(granularity)
    cpu_offload(model,execution_device=execution_device,**options)
    return model
