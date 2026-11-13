"""Audit real Qwen weight placement before inference.

Read-only, avoids allocating/copying weights or materializing meta tensors.
The HF device map can designate a parent of a parameter, so use longest
matching module prefix for effective placement.
"""
from __future__ import annotations
from collections import defaultdict


def module_device(name, device_map):
    if not isinstance(device_map, dict) or not device_map:
        return "unknown"
    matched=[(len(prefix),device) for prefix,device in device_map.items()
             if not prefix or name==prefix or name.startswith(prefix+".")]
    return str(max(matched,key=lambda item:item[0])[1]) if matched else "unknown"


def audit_placement(model, *, top_k=12):
    mapping=getattr(model,"hf_device_map",None)
    if not isinstance(mapping,dict) or not mapping:
        return {"status":"no_device_map","devices":{},"large_cpu_modules":[]}
    sizes=defaultdict(int)
    device_totals=defaultdict(int)
    identities=set()
    for name,param in model.named_parameters(remove_duplicate=False):
        identity=id(param)
        if identity in identities:
            continue
        identities.add(identity)
        nbytes=param.numel()*param.element_size()
        device=module_device(name,mapping)
        device_totals[device]+=nbytes
        category=("decoder" if any(token in name.lower() for token in
                  ("text_model","text_decoder","thinker.model","language_model"))
                  else "audio" if "audio" in name.lower() else "other")
        sizes[(name.rsplit(".",1)[0],device,category)]+=nbytes
    ranked=sorted(((n,d,c,b) for (n,d,c),b in sizes.items()
                   if d in ("cpu","disk")),key=lambda x:x[3],reverse=True)
    return {"status":"measured","devices":{
        key:round(value/1024**2,2) for key,value in device_totals.items()},
        "large_cpu_modules":[{"name":n,"device":d,"category":c,
                              "weight_mib":round(b/1024**2,2)}
                             for n,d,c,b in ranked[:top_k]],
        "cuda_decoder_module_count":sum(1 for (n,d,c) in sizes if c=="decoder"
                                        and d in ("0","cuda","cuda:0")),
        "cpu_decoder_module_count":sum(1 for (n,d,c) in sizes if c=="decoder"
                                       and d=="cpu")}
