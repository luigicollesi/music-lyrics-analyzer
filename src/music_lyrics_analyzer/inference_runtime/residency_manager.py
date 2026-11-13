"""Safe GPU weight-residency planning for Accelerate device_map auto.

Accelerate maps whole modules to GPU/CPU; CUDA modules remain resident
between tokens while CPU modules are moved on demand. No private hooks
or hidden CUDA allocation promises. Budget applies to model placement
only, not audio activations or KV cache.
"""
from __future__ import annotations


def budget_for_residency(*, physical_vram_gib, weights_gib,
                         reserve_gib=1.5, cpu_memory_gib=12.):
    if physical_vram_gib<=0:
        raise ValueError("GPU residency requires CUDA")
    if reserve_gib<=0 or cpu_memory_gib<1:
        raise ValueError("Invalid GPU reserve or CPU memory budget")
    if not .25<=weights_gib<physical_vram_gib:
        raise ValueError("GPU weight budget must be at least 0.25 GiB and below VRAM")
    if weights_gib+reserve_gib>physical_vram_gib:
        raise ValueError("GPU weight budget plus activation reserve exceeds physical VRAM")
    return {"max_memory":{0:f"{int(weights_gib*1024)}MiB",
                           "cpu":f"{int(cpu_memory_gib*1024)}MiB"},
            "device_map":"auto","low_cpu_mem_usage":True}


def residency_summary(device_map):
    if not isinstance(device_map,dict):
        return {"status":"unknown","cuda_modules":None,"cpu_modules":None}
    result={"status":"observed","cuda_modules":0,"cpu_modules":0,"disk_modules":0}
    for device in device_map.values():
        if device in (0,"cuda","cuda:0"):
            result["cuda_modules"]+=1
        elif str(device)=="cpu":
            result["cpu_modules"]+=1
        elif str(device)=="disk":
            result["disk_modules"]+=1
    return result
