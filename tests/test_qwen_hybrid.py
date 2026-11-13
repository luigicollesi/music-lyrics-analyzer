"""Hybrid parameter validation without model downloads."""
import numpy as np
import pytest
from music_lyrics_analyzer.qwen_backend import transcribe_qwen_full


def test_unknown_model_is_rejected_in_hybrid_mode():
    with pytest.raises(ValueError,match="Unknown Qwen"):
        transcribe_qwen_full(np.zeros(16000,dtype=np.float32),
                             model_size="unknown",device="hybrid")
