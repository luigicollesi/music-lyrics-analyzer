"""Offline unit tests for memory-managed runtime boundaries."""
import pytest
from music_lyrics_analyzer.inference_runtime.config import RuntimeConfig
from music_lyrics_analyzer.inference_runtime.memory_profiler import MemoryProfiler
from music_lyrics_analyzer.inference_runtime.device_manager import validate_offload_hooks

class CudaStub:
    def is_available(self):return False

class TorchStub:
    cuda=CudaStub()

class Model:
    def __init__(self,meta=False,hook=False):
        self.meta=meta
        if hook:self._hf_hook=object()
    def named_parameters(self):
        return iter([("layer.weight",type("P",(),{"is_meta":self.meta})())])
    def modules(self):
        return [self]

def test_cpu_profiler_collects_checkpoints():
    p=MemoryProfiler(TorchStub())
    p.mark("start")
    assert p.to_dict()["checkpoints"][0]["phase"]=="start"

def test_profiler_can_be_disabled():
    p=MemoryProfiler(TorchStub(),enabled=False)
    p.mark("ignored")
    assert p.to_dict()["checkpoints"]==[]

def test_orphan_meta_parameters_are_rejected():
    with pytest.raises(RuntimeError,match="without Accelerate"):
        validate_offload_hooks(Model(meta=True))

def test_offloaded_meta_parameters_are_allowed_with_hooks():
    assert validate_offload_hooks(Model(meta=True,hook=True))["meta_present"]

def test_config_rejects_non_boolean_flags():
    with pytest.raises(ValueError):
        RuntimeConfig(profile="yes")
