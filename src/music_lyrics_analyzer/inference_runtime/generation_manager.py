"""Generation-state diagnostics.

This phase does not change Transformers cache implementations or generation
semantics. Add compatible cache strategies only after profiling.
"""
def generation_policy():
    return "baseline"
