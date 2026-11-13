"""Memory budgets and device-map summaries for Qwen full ASR."""
import pytest
from music_lyrics_analyzer.inference_runtime.residency_manager import (
    budget_for_residency,residency_summary)


def test_valid_residency_budget_is_bounded():
    p=budget_for_residency(physical_vram_gib=4,weights_gib=1,
                           reserve_gib=1.5,cpu_memory_gib=12)
    assert p["device_map"]=="auto"
    assert p["max_memory"]=={0:"1024MiB","cpu":"12288MiB"}


@pytest.mark.parametrize("weights,reserve",[(4,1),(.1,1),(3,1.5),(1,0)])
def test_invalid_vram_budget_fails(weights,reserve):
    with pytest.raises(ValueError):
        budget_for_residency(physical_vram_gib=4,
                             weights_gib=weights,reserve_gib=reserve)


def test_residency_summary_counts_device_placement():
    s=residency_summary({"model.embed":0,"model.layer.0":"cuda:0",
                         "model.layer.1":"cpu","model.layer.2":"disk"})
    assert s=={"status":"observed","cuda_modules":2,
               "cpu_modules":1,"disk_modules":1}
