"""Regression: Accelerate meta placeholders must not become ASR input devices."""
from types import SimpleNamespace
from music_lyrics_analyzer.qwen_backend import _QwenOffloadDeviceProxy


class OffloadedModel:
    device="meta"
    dtype="float16"
    hf_device_map={"layers.0":"cpu"}

    def generate(self, **kwargs):
        return kwargs["input_ids"]


def test_offload_proxy_exposes_actual_execution_device():
    model=OffloadedModel()
    proxy=_QwenOffloadDeviceProxy(model,"cuda:0","float16")
    assert proxy.device=="cuda:0"
    assert proxy.dtype=="float16"
    assert proxy.hf_device_map=={"layers.0":"cpu"}
    assert proxy.generate(input_ids="data")=="data"
    assert model.device=="meta"
