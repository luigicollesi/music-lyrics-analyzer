"""Utilities to validate Qwen wrappers before offloaded generation."""
from .device_manager import validate_offload_hooks

def validate_model_for_generation(model):
    return validate_offload_hooks(model)
