"""Tell Transformers generation the real execution device of a dispatched model.

Accelerate CPU offload deliberately leaves parameters on meta. The inherited
PreTrainedModel.device property can report meta if the model's first
parameter is CPU-offloaded, even when its inputs and decoder run on CUDA.
Transformers generation creates special-token tensors using self.device in
some versions; meta token scalars fail at Tensor.item().
"""
from __future__ import annotations


def bind_generation_execution_device(model, execution_device):
    """Override only the reported device, not parameters or Accelerate hooks.

    A per-instance dynamic subclass avoids globally modifying upstream Qwen
    classes. Preserve all existing methods, MRO, and module parameters.
    """
    import torch
    device=torch.device(execution_device)
    if device.type=="meta":
        raise ValueError("Execution device cannot be meta.")
    base=type(model)
    bound_type=type(
        f"{base.__name__}ExecutionDevice",
        (base,),
        {"device":property(lambda self: device),
         "__module__":base.__module__},
    )
    try:
        model.__class__=bound_type
    except TypeError as exc:
        raise RuntimeError("Cannot bind Qwen generation execution device") from exc
    if model.device!=device:
        raise RuntimeError("Qwen execution device binding failed")
    return model
