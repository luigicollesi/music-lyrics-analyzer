"""Activation management contracts.

Audio-encoder activations are not automatically bounded by CPU weight
offloading. Architecture-specific attention profiling is required before
any memory optimization can safely claim equivalence.
"""
def activation_policy():
    return "unmanaged"
