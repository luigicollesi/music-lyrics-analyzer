"""CI-safe structural MMS checks (never claim acoustic/weight validation)."""
from types import SimpleNamespace

import pytest

from music_lyrics_analyzer.model_integrity import inspect_mms


class FakeTokenizer:
    pad_token_id = 0
    unk_token_id = 3

    def __init__(self, vocab=None):
        self.vocab = vocab or {"<pad>": 0, "a": 1, "b": 2, "<unk>": 3}

    def get_vocab(self):
        return self.vocab


def model(size=4):
    return SimpleNamespace(lm_head=SimpleNamespace(out_features=size))


def test_matching_portuguese_head_and_tokenizer_is_only_structurally_valid():
    result = inspect_mms(
        model(), FakeTokenizer(), model_id="facebook/mms-1b-all",
        language="por", adapter_requested=True)
    assert result.status == "structurally_compatible"
    assert result.output_size == result.vocabulary_size == 4
    assert result.blank_id == 0
    assert result.adapter_requested is True
    assert "trained weights" in " ".join(result.warnings)
    assert result.to_dict()["language"] == "por"


def test_mismatched_vocabulary_index_fails_before_inference():
    with pytest.raises(ValueError, match="exceeds output size"):
        inspect_mms(
            model(), FakeTokenizer({"<pad>": 0, "a": 1, "<unk>": 6}),
            model_id="x", language="por", adapter_requested=True)


def test_missing_blank_token_fails_before_inference():
    tokenizer = FakeTokenizer({"a": 1, "b": 2, "<unk>": 3})
    with pytest.raises(ValueError, match="blank/pad"):
        inspect_mms(model(), tokenizer, model_id="x",
                    language="por", adapter_requested=True)


def test_missing_adapter_is_a_warning_not_false_success():
    audit=inspect_mms(model(), FakeTokenizer(), model_id="x",
                      language="por", adapter_requested=False)
    assert any("not requested" in w for w in audit.warnings)


def test_head_extra_dimensions_are_reported():
    audit=inspect_mms(model(5), FakeTokenizer(), model_id="x",
                      language="por", adapter_requested=True)
    assert audit.output_size == 5
    assert any("CTC head has 5" in w for w in audit.warnings)


def test_candidate_json_uses_actual_newline_not_literal_backslash_n():
    # Avoid loading MMS in CI; guard against prior bad export expression.
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1]
              / "src/music_lyrics_analyzer/cli.py").read_text()
    marker = source.index("candidate_path.write_text(")
    expression = source[marker:source.index("print(f'  CTC candidatos:", marker)]
    assert "indent=2)+'\\n'" in expression
    assert "indent=2)+'\\\\n'" not in expression
