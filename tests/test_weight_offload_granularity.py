"""Check optional per-layer Qwen offloading without downloading a model."""
import pytest
from music_lyrics_analyzer.inference_runtime.weight_manager import offload_options


def test_tensor_granularity_is_the_unchanged_default():
    assert offload_options("tensor")=={}


def test_layer_granularity_only_preloads_text_decoder_blocks():
    assert offload_options("layer")=={
        "preload_module_classes":["Qwen3ASRThinkerTextDecoderLayer"]}


def test_invalid_granularity_is_rejected_before_model_load():
    with pytest.raises(ValueError,match="granularity"):
        offload_options("decoder")
