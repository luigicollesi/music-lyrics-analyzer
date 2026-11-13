"""Explicitly distinguish static placement from dynamic parameter offloading."""
def attach_weight_offload(model, execution_device):
    from accelerate import cpu_offload
    cpu_offload(model, execution_device=execution_device)
    return model
