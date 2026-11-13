"""Placement audits must not materialize offloaded parameters."""
from types import SimpleNamespace
from music_lyrics_analyzer.inference_runtime.placement_audit import (
    audit_placement,module_device)

class Parameter:
    def __init__(self,n,dtype_bytes=2):
        self.n=n
        self.bytes=dtype_bytes
    def numel(self):return self.n
    def element_size(self):return self.bytes

class Model:
    hf_device_map={"":"cpu","thinker.text_model.layers.0":"cuda:0"}
    def named_parameters(self,remove_duplicate=False):
        p=Parameter(128)
        yield "thinker.text_model.layers.0.weight",p
        yield "thinker.text_model.layers.1.weight",Parameter(64)
        yield "thinker.text_model.layers.0.alias",p

def test_effective_device_uses_longest_prefix():
    assert module_device("thinker.text_model.layers.0.weight",Model.hf_device_map)=="cuda:0"
    assert module_device("thinker.text_model.layers.1.weight",Model.hf_device_map)=="cpu"

def test_audit_deduplicates_tied_parameters():
    audit=audit_placement(Model())
    assert audit["status"]=="measured"
    assert audit["devices"]["cuda:0"]==round(256/1024**2,2)
    assert audit["cpu_decoder_module_count"]==1

def test_missing_device_map_is_explicit():
    assert audit_placement(SimpleNamespace())["status"]=="no_device_map"
