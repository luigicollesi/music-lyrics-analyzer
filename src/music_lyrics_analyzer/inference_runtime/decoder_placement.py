"""Conservative explicit Qwen decoder block placement.

Plan from a CPU-loaded model without moving tensors. Accelerate dispatch
then preserves selected text decoder blocks on CUDA across generated tokens.
No private hooks or allocation of activation buffers at planning time.
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
    assigned={}
    selected_bytes=0
    # Fully resident transformer layers only; no partial layer splits.
    for name,size in candidates:
        if selected_bytes+size<=budget:
            assigned[name]=0
            selected_bytes+=size
    if not assigned:
        raise RuntimeError("Budget insufficient for one full Qwen text decoder layer.")
    return {"":"cpu",**assigned}, {
        "strategy":"decoder_priority",
        "resident_decoder_layers":list(assigned),
        "resident_weight_mib":round(selected_bytes/1024**2,2),
        "total_decoder_layers":len(candidates),
        "budget_mib":round(budget/1024**2,2),
    }


def apply_decoder_priority(model, *, gpu_weights_gib, reserve_gib,
                           physical_vram_gib):
    from accelerate import dispatch_model
    mapping,summary=plan_decoder_priority(
        model,gpu_weights_gib=gpu_weights_gib,
        reserve_gib=reserve_gib,physical_vram_gib=physical_vram_gib)
    # dispatch_model handles CPU offload hooks and materialization.
    dispatched=dispatch_model(model,device_map=mapping,
                              main_device="cuda:0",offload_buffers=False)
    return dispatched,summary
