"""Full-song ASR checks without downloading Qwen or initializing CUDA."""
import numpy as np
import pytest
from music_lyrics_analyzer.qwen_backend import transcribe_qwen_full


def test_rejects_invalid_token_limit_before_model_import():
    with pytest.raises(ValueError,match="max_new_tokens"):
        transcribe_qwen_full(np.zeros(32000,dtype=np.float32),
                             max_new_tokens=32)


def test_rejects_audio_exceeding_configured_length_before_loading():
    with pytest.raises(ValueError,match="safety duration"):
        transcribe_qwen_full(np.zeros(48000,dtype=np.float32),
                             max_audio_seconds=2.)


def test_rejects_empty_audio_before_model_loading():
    with pytest.raises(ValueError,match="mono audio"):
        transcribe_qwen_full(np.zeros(100,dtype=np.float32))


def test_rejects_unknown_model_before_model_loading():
    with pytest.raises(ValueError,match="Unknown Qwen"):
        transcribe_qwen_full(np.zeros(16000,dtype=np.float32),model_size="bad")
