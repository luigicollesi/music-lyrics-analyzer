"""Lightweight configuration for experiments, with no model imports."""
from dataclasses import dataclass

@dataclass(frozen=True)
class RuntimeConfig:
    profile: bool = False
    strict_meta_validation: bool = True

    def __post_init__(self):
        if not isinstance(self.profile,bool) or not isinstance(self.strict_meta_validation,bool):
            raise ValueError("Runtime flags must be booleans.")
