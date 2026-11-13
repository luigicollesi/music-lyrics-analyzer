"""Qwen3 ASR device selection stays testable without GPU or model download."""
from types import SimpleNamespace
import pytest
from music_lyrics_analyzer.qwen_backend import choose_device, MODELS

class CUDA:
    def __init__(self,active=True,arch=(6,1),arches=None,memory=4):
        self.active=active;self.arch=arch;self.arches=arches or ["sm_75"]
        self.memory=memory
    def is_available(self):return self.active
    def get_device_capability(self,index):return self.arch
    def get_device_name(self,index):return "GTX 1050 Ti"
    def get_device_properties(self,index):
        return SimpleNamespace(total_memory=self.memory*1024**3)
    def get_arch_list(self):return self.arches

def fake_cuda(cuda):
    return SimpleNamespace(cuda=cuda)

def test_cuda_rejects_incompatible_pascal_wheel():
    with pytest.raises(RuntimeError,match="sm_61"):
        choose_device(fake_cuda(CUDA()),"cuda")

def test_auto_falls_back_explicitly_to_cpu():
    plan=choose_device(fake_cuda(CUDA()),"auto")
    assert plan.device=="cpu" and plan.dtype=="float32"

def test_cpu_is_always_available():
    plan=choose_device(fake_cuda(CUDA(active=False)),"cpu")
    assert plan.device=="cpu"

def test_cuda_uses_fp16_on_compatible_kernel():
    plan=choose_device(fake_cuda(CUDA(arches=["sm_61"])),"cuda")
    assert plan.device=="cuda:0" and plan.dtype=="float16"

def test_cuda_fails_insufficient_memory_budget():
    with pytest.raises(RuntimeError,match="VRAM insuficiente"):
        choose_device(fake_cuda(CUDA(arches=["sm_61"],memory=4)),
                      "cuda",minimum_vram_gb=5)

def test_model_names_are_explicit():
    assert MODELS["1.7b"]=="Qwen/Qwen3-ASR-1.7B"
