"""Regression tests for temporary Accelerate transfer instrumentation."""
import sys
from types import SimpleNamespace
from music_lyrics_analyzer.inference_runtime.transfer_profiler import TransferProfiler

class Tensor:
    def numel(self):return 16
    def element_size(self):return 2

def test_transfer_hook_is_restored_and_counts_calls(monkeypatch):
    original_calls=[]
    def transfer(module,name,device,value=None):
        original_calls.append((name,device))
        return "ok"
    hooks=SimpleNamespace(set_module_tensor_to_device=transfer)
    monkeypatch.setitem(sys.modules,"accelerate",SimpleNamespace(hooks=hooks))
    monkeypatch.setitem(sys.modules,"accelerate.hooks",hooks)
    tracer=TransferProfiler(log_every=100)
    with tracer.observe():
        assert hooks.set_module_tensor_to_device(None,"weight","cuda",Tensor())=="ok"
        assert tracer.calls==1
        assert tracer.estimated_bytes==32
    assert hooks.set_module_tensor_to_device is transfer

def test_transfer_hook_restored_on_exception(monkeypatch):
    original=lambda *a,**kw:None
    hooks=SimpleNamespace(set_module_tensor_to_device=original)
    monkeypatch.setitem(sys.modules,"accelerate",SimpleNamespace(hooks=hooks))
    monkeypatch.setitem(sys.modules,"accelerate.hooks",hooks)
    try:
        with TransferProfiler().observe():
            raise RuntimeError("failed")
    except RuntimeError:
        pass
    assert hooks.set_module_tensor_to_device is original
