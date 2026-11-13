"""Portable CPU/CUDA memory profiling with explicit phase checkpoints."""
from __future__ import annotations
import os
import time

class MemoryProfiler:
    def __init__(self, torch, *, enabled=True):
        self.torch=torch
        self.enabled=enabled
        self.start=time.monotonic()
        self.checkpoints=[]

    def mark(self, phase):
        if not self.enabled:return
        info={"phase":phase,"elapsed_seconds":round(time.monotonic()-self.start,3)}
        try:
            import resource
            # ru_maxrss is KiB on Linux (documented target).
            info["process_peak_rss_gb"]=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,3)
        except (ImportError,OSError):
            pass
        if self.torch.cuda.is_available():
            info["cuda_allocated_gb"]=round(self.torch.cuda.memory_allocated(0)/1024**3,3)
            info["cuda_reserved_gb"]=round(self.torch.cuda.memory_reserved(0)/1024**3,3)
            info["cuda_peak_allocated_gb"]=round(self.torch.cuda.max_memory_allocated(0)/1024**3,3)
            free,total=self.torch.cuda.mem_get_info(0)
            info["cuda_free_gb"]=round(free/1024**3,3)
            info["cuda_total_gb"]=round(total/1024**3,3)
        self.checkpoints.append(info)

    def to_dict(self):
        return {"checkpoints":self.checkpoints}
