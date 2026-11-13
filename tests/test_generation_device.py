"""A CUDA execution-device facade must not materialize offloaded weights."""
from types import SimpleNamespace
import pytest
from music_lyrics_analyzer.inference_runtime.generation_device import (
    bind_generation_execution_device)


def test_dispatched_model_generation_device_is_cuda_without_mutating_weights():
    torch=pytest.importorskip("torch")
    class SmallModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight=torch.nn.Parameter(torch.empty(1,device="meta"))
        @property
        def device(self):
            return self.weight.device
        def special_token(self):
            return torch.tensor(2,device=self.device)
    model=SmallModel()
    assert model.device.type=="meta"
    original_parameter=model.weight
    bind_generation_execution_device(model,"cpu")
    assert model.device.type=="cpu"
    assert model.special_token().item()==2
    assert model.weight is original_parameter
    assert model.weight.device.type=="meta"
    assert SmallModel().device.type=="meta"


def test_reject_meta_execution_target():
    torch=pytest.importorskip("torch")
    with pytest.raises(ValueError,match="meta"):
        bind_generation_execution_device(torch.nn.Linear(1,1),"meta")
