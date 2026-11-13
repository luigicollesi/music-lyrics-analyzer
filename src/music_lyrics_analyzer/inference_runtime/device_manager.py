"""Inspect an Accelerate-offloaded model without materializing its weights."""
from __future__ import annotations

def inspect_meta(model, *, limit=12):
    paths=[]
    for name,param in model.named_parameters():
        if getattr(param,"is_meta",False):
            paths.append(name)
            if len(paths)>=limit:break
    return {"meta_parameter_examples":paths,
            "meta_present":bool(paths),
            "accelerate_hooks":bool(getattr(model,"_hf_hook",None))}

def validate_offload_hooks(model):
    summary=inspect_meta(model)
    if summary["meta_present"] and not summary["accelerate_hooks"]:
        # Hooks can be attached to child modules only: inspect recursively.
        hooked=any(getattr(child,"_hf_hook",None) is not None
                   for child in model.modules())
        if not hooked:
            raise RuntimeError("Model has meta parameters without Accelerate offload hooks.")
    return summary
