"""Regression: Qwen delegates outer generate to thinker.generate."""
import pytest
from music_lyrics_analyzer.inference_runtime.generation_device import bind_generation_execution_device

def test_inner_thinker_has_independent_execution_device():
    torch=pytest.importorskip("torch")
    class Thinker(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight=torch.nn.Parameter(torch.empty(1,device="meta"))
        @property
        def device(self):
            return self.weight.device
        def generate(self):
            return torch.tensor(151645,device=self.device).item()
    class Outer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.thinker=Thinker()
        @property
        def device(self):
            return next(self.parameters()).device
        def generate(self):
            return self.thinker.generate()
    model=Outer()
    bind_generation_execution_device(model,"cpu")
    assert model.device.type=="cpu"
    assert model.thinker.device.type=="meta"
    bind_generation_execution_device(model.thinker,"cpu")
    assert model.generate()==151645
    assert model.thinker.weight.device.type=="meta"
