"""Smoke-test the real CTC decoder without downloading a neural model."""
import numpy as np
import pytest

from music_lyrics_analyzer.ctc_backend import segment_emissions


def test_real_ctc_segmentation_backtrack_has_vocabulary():
    pytest.importorskip("ctc_segmentation")

    class Tokenizer:
        pad_token_id = 0
        unk_token_id = 4

        def get_vocab(self):
            return {"<pad>": 0, "a": 1, "b": 2, "c": 3, "<unk>": 4}

        def __call__(self, value, *, add_special_tokens=False):
            return type("Tokens", (), {
                "input_ids": [self.get_vocab().get(ch, 4) for ch in value]
            })()

    # Frames follow a -> blank -> b -> blank -> c -> blank.
    ids = [0]*7 + [1]*7 + [0]*4 + [2]*7 + [0]*4 + [3]*7 + [0]*7
    logits = np.full((len(ids), 5), -8.0, dtype=np.float32)
    for frame, token in enumerate(ids):
        logits[frame, token] = -0.01
    diagnostic = {}
    candidates = segment_emissions(
        logits, [{"id": "one", "text": "ab"},
                 {"id": "two", "text": "c"}],
        Tokenizer(), duration=len(ids)*0.02,
        threshold=-100, diagnostic=diagnostic)
    assert diagnostic["status"] in {"ok", "all_rejected"}
    assert diagnostic.get("status") != "invalid_emissions"
    assert isinstance(candidates, list)
