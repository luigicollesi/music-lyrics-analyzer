"""Explicit, non-overlapping Qwen decoder GPU placement.

Build a complete device map with no CPU ancestor of a CUDA descendant.
Validate that weights were materialized before calling Accelerate dispatch.
"""
from __future__ import annotations


def decoder_layer_candidates(model):
    result=[]
    for name,module in model.named_modules():
        if type(module).__name__!="Qwen3ASRThinkerTextDecoderLayer":
            continue
        size=sum(p.numel()*p.element_size() for p in module.parameters(recurse=True))
        if size:
            result.append((name,size))
    return result


def _partition_map(model, selected):
    """Cover all parameter owners without overlapping device-map prefixes."""
    mapping={}
    def walk(module,prefix):
        # Entire selected decoder layers are indivisible GPU islands.
        if prefix in selected:
            mapping[prefix]=0
            return
        children=list(module.named_children())
        direct_params=list(module.named_parameters(recurse=False))
        direct_buffers=list(module.named_buffers(recurse=False))
        has_cuda_descendant=any(p.startswith(prefix+".") if prefix else True
                                for p in selected)
        if not has_cuda_descendant:
            mapping[prefix]="cpu"
            return
        # Direct parameters must receive their own entries if splitting
        # a module's children. Device maps accept parameter/buffer names.
        for name,_ in direct_params+direct_buffers:
            mapping[f"{prefix}.{name}" if prefix else name]="cpu"
        if not children and prefix:
            mapping[prefix]="cpu"
        for child_name,child in children:
            path=f"{prefix}.{child_name}" if prefix else child_name
            walk(child,path)
    walk(model,"")
    return mapping


def plan_decoder_priority(model, *, gpu_weights_gib, reserve_gib,
                          physical_vram_gib, safety_factor=0.9):
    if gpu_weights_gib<=0 or reserve_gib<=0 or (
        gpu_weights_gib+reserve_gib>physical_vram_gib):
        raise ValueError("GPU weights plus reserve exceed device VRAM")
    candidates=decoder_layer_candidates(model)
    if not candidates:
        raise RuntimeError("Cannot identify Qwen text decoder layers; "
                           "refusing unsafe device placement.")
    budget=int(gpu_weights_gib*1024**3*safety_factor)
    assigned=[]
    selected_bytes=0
    for name,size in candidates:
        if selected_bytes+size<=budget:
            assigned.append(name)
            selected_bytes+=size
    if not assigned:
        raise RuntimeError("Budget insufficient for one full Qwen text decoder layer.")
    mapping=_partition_map(model,set(assigned))
    return mapping,{
        "strategy":"decoder_priority",
        "resident_decoder_layers":assigned,
        "resident_weight_mib":round(selected_bytes/1024**2,2),
        "total_decoder_layers":len(candidates),
        "budget_mib":round(budget/1024**2,2),
        "device_map_entries":len(mapping),
    }


def apply_decoder_priority(model, *, gpu_weights_gib, reserve_gib,
                           physical_vram_gib):
    from accelerate import dispatch_model
    from accelerate.utils import check_device_map
    mapping,summary=plan_decoder_priority(
        model,gpu_weights_gib=gpu_weights_gib,
        reserve_gib=reserve_gib,physical_vram_gib=physical_vram_gib)
    check_device_map(model,mapping)
    # A meta tensor without a backing checkpoint value cannot be restored
    # by dispatch_model. Detect this before mutating any model hooks.
    missing=[name for name,p in model.named_parameters()
             if p.device.type=="meta"]
    if missing:
        raise RuntimeError(
            "Qwen decoder placement requires materialized CPU weights; "
            f"{len(missing)} parameters are meta (first: {missing[0]}). "
            "Load the full model on CPU before explicit dispatch.")
    dispatched=dispatch_model(model,device_map=mapping,
                              main_device="cuda:0",offload_buffers=False)
    return dispatched,summary
