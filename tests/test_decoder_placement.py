"""Decoder first planning, without a GPU or downloading models."""
import pytest
from music_lyrics_analyzer.inference_runtime.decoder_placement import (
    plan_decoder_priority,decoder_layer_candidates)

class Param:
    def __init__(self, count):self.count=count
    def numel(self):return self.count
    def element_size(self):return 2

class Qwen3ASRThinkerTextDecoderLayer:
    def __init__(self, mib):self.mib=mib
    def parameters(self, recurse=True):
        return [Param(self.mib*1024*1024//2)]

class Root:
    def __init__(self,layers):
        self.layers=layers
    def named_modules(self):
        yield "",self
        for i,layer in enumerate(self.layers):
            yield f"thinker.text_model.layers.{i}",layer

def test_decoder_budget_keeps_only_whole_layers():
    model=Root([Qwen3ASRThinkerTextDecoderLayer(300) for _ in range(4)])
    mapping,summary=plan_decoder_priority(model,gpu_weights_gib=1,
                                          reserve_gib=1.5,physical_vram_gib=4)
    assert mapping[""]=="cpu"
    assert len(summary["resident_decoder_layers"])==3
    assert summary["resident_weight_mib"]==900

def test_refuses_unknown_decoder():
    with pytest.raises(RuntimeError,match="Cannot identify"):
        plan_decoder_priority(Root([]),gpu_weights_gib=1,
                              reserve_gib=1.5,physical_vram_gib=4)

def test_rejects_oversubscribed_budget():
    with pytest.raises(ValueError):
        plan_decoder_priority(Root([]),gpu_weights_gib=3,
                              reserve_gib=1.5,physical_vram_gib=4)
