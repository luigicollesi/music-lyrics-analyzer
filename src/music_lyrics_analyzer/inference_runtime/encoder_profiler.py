"""Opt-in, bounded-memory phase sampling of Qwen audio encoder.

Hooks observe runtime CUDA memory around module forwards. They neither
change inputs nor retain outputs, and always detach in finally.
"""
from __future__ import annotations
from contextlib import contextmanager
from time import monotonic


def locate_audio_modules(model):
    """Find audio encoder through known Qwen model wrappers."""
    candidates=[model]
    for path in ("model","thinker","model.thinker","model.model",
                 "model.audio_tower","model.model.audio_tower"):
        node=model
        for part in path.split("."):
            node=getattr(node,part,None)
            if node is None:break
        if node is not None:candidates.append(node)
    seen=set()
    for root in candidates:
        if id(root) in seen:continue
        seen.add(id(root))
        for name in ("audio_tower","audio_encoder"):
            encoder=getattr(root,name,None)
            if encoder is not None and hasattr(encoder,"register_forward_hook"):
                return encoder
    return None


class EncoderMemoryTrace:
    def __init__(self, torch, *, max_events=160, synchronize=False):
        if max_events<1:raise ValueError("max_events must be positive")
        self.torch=torch
        self.max_events=max_events
        self.synchronize=synchronize
        self.events=[]
        self.truncated=0
        self.started=monotonic()
        self._active={}

    def _sample(self, module, name, phase):
        if len(self.events)>=self.max_events:
            self.truncated+=1
            return
        if self.torch.cuda.is_available():
            if self.synchronize:self.torch.cuda.synchronize()
            allocated=self.torch.cuda.memory_allocated()
            reserved=self.torch.cuda.memory_reserved()
            peak=self.torch.cuda.max_memory_allocated()
            event={"name":name,"phase":phase,"elapsed_seconds":
                   round(monotonic()-self.started,3),
                   "allocated_mib":round(allocated/1024**2,1),
                   "reserved_mib":round(reserved/1024**2,1),
                   "peak_allocated_mib":round(peak/1024**2,1)}
            if phase=="post":
                baseline=self._active.pop(id(module),None)
                if baseline is not None:
                    event["delta_allocated_mib"]=round((allocated-baseline)/1024**2,1)
            else:
                self._active[id(module)]=allocated
            self.events.append(event)

    @contextmanager
    def observe(self, model):
        encoder=locate_audio_modules(model)
        if encoder is None:
            yield {"status":"audio_encoder_not_found","events":self.events}
            return
        hooks=[]
        try:
            targets=[("audio_encoder",encoder)]
            layers=getattr(encoder,"layers",None)
            if layers is not None:
                for index,layer in enumerate(layers):
                    targets.append((f"audio_encoder.layers.{index}",layer))
            for name,module in targets:
                hooks.append(module.register_forward_pre_hook(
                    lambda mod,args,n=name:self._sample(mod,n,"pre")))
                hooks.append(module.register_forward_hook(
                    lambda mod,args,output,n=name:self._sample(mod,n,"post")))
            yield {"status":"instrumented","events":self.events}
        finally:
            for hook in hooks:hook.remove()
            self._active.clear()

    def to_dict(self):
        return {"events":self.events,"truncated_event_count":self.truncated,
                "synchronize":self.synchronize,
                "status":"sampled" if self.events else "no_events"}
