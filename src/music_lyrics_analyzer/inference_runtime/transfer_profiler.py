"""Opt-in accounting for Accelerate host-side weight materialization.

We wrap the exact function reference imported by accelerate.hooks, restore
it in finally, and never retain tensor objects. CPU wall time measures
Python dispatch / enqueue overhead, not synchronized PCIe latency.
"""
from contextlib import contextmanager
from time import perf_counter
import sys


class TransferProfiler:
    def __init__(self, *, log_every=1000):
        if log_every<1:
            raise ValueError("log_every must be positive")
        self.calls=0
        self.estimated_bytes=0
        self.host_seconds=0.0
        self.log_every=log_every

    def report(self):
        return {"calls":self.calls,
                "estimated_gib":round(self.estimated_bytes/1024**3,3),
                "host_seconds":round(self.host_seconds,3),
                "measurement":"host_wall_time_not_synchronized_cuda"}

    @contextmanager
    def observe(self):
        import accelerate.hooks as hooks
        original=hooks.set_module_tensor_to_device
        def wrapped(*args,**kwargs):
            value=kwargs.get("value")
            if value is None and len(args)>3:
                value=args[3]
            size=(value.numel()*value.element_size()
                  if value is not None and hasattr(value,"numel") else 0)
            start=perf_counter()
            try:
                return original(*args,**kwargs)
            finally:
                self.calls+=1
                self.estimated_bytes+=size
                self.host_seconds+=perf_counter()-start
                if self.calls%self.log_every==0:
                    print("[Qwen transfer profile] "+str(self.report()),
                          file=sys.stderr,flush=True)
        hooks.set_module_tensor_to_device=wrapped
        try:
            yield self
        finally:
            hooks.set_module_tensor_to_device=original
