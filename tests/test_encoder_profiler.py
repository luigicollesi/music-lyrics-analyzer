"""Encoder traces are observational and hooks do not survive inference."""
from types import SimpleNamespace
from music_lyrics_analyzer.inference_runtime.encoder_profiler import (
    EncoderMemoryTrace,locate_audio_modules)

class Hook:
    def __init__(self,fn):self.fn=fn
    def remove(self):self.fn()

class Module:
    def __init__(self):self.hooks=[];self.layers=[]
    def register_forward_pre_hook(self,fn):
        self.hooks.append(fn)
        return Hook(lambda:self.hooks.remove(fn))
    def register_forward_hook(self,fn):
        self.hooks.append(fn)
        return Hook(lambda:self.hooks.remove(fn))

class CUDA:
    def is_available(self):return False

def test_locate_nested_qwen_audio_tower():
    encoder=Module()
    wrapper=SimpleNamespace(model=SimpleNamespace(thinker=SimpleNamespace(audio_tower=encoder)))
    assert locate_audio_modules(wrapper) is encoder

def test_hooks_detached_on_success_and_failure():
    encoder=Module()
    model=SimpleNamespace(audio_tower=encoder)
    trace=EncoderMemoryTrace(SimpleNamespace(cuda=CUDA()))
    with trace.observe(model) as status:
        assert status["status"]=="instrumented"
        assert len(encoder.hooks)==2
    assert not encoder.hooks
    try:
        with trace.observe(model):
            raise RuntimeError("synthetic")
    except RuntimeError:
        pass
    assert not encoder.hooks

def test_missing_encoder_is_reported_without_faking_results():
    trace=EncoderMemoryTrace(SimpleNamespace(cuda=CUDA()))
    with trace.observe(object()) as status:
        assert status["status"]=="audio_encoder_not_found"
    assert trace.to_dict()["status"]=="no_events"
