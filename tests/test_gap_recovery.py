"""Unit tests for bounded acoustic retries (no model download)."""
import numpy as np

from music_lyrics_analyzer.core import parse_lyrics
from music_lyrics_analyzer.gap_recovery import retry_unresolved_gaps


class Word:
    def __init__(self, text, start, end):
        self.word, self.start, self.end = text, start, end
        self.probability = 0.9


class Segment:
    def __init__(self, words):
        self.words = words


class Model:
    def __init__(self):
        self.calls = []

    def align(self, audio, text, *, language, original_split):
        self.calls.append(text)
        tokens = text.split()
        size = len(audio) / 16000
        class Result:
            pass
        result = Result()
        result.segments = [Segment([
            Word(t, size * k / len(tokens) + .03,
                 min(size - .01, size * (k + 1) / len(tokens) - .03))
            for k, t in enumerate(tokens)
        ])]
        return result


def test_internal_missing_words_retried_with_actual_lyric_text():
    words, _ = parse_lyrics("hoje como mingau para comer")
    words[0].update(start=1.0, end=1.3)
    words[3].update(start=3.5, end=3.9)
    words[4].update(start=4.0, end=4.3)
    model = Model()
    report = retry_unresolved_gaps(model, np.zeros(16000 * 7),
                                   words, .5, 5.0, "pt")
    assert model.calls == ["como mingau"]
    assert report["recovered"] == 2
    assert words[1]["start"] >= 1.3
    assert words[2]["end"] <= 3.5
    assert words[1]["alignment"]["method"] == "stable-ts-gap-retry"


def test_long_edge_window_is_not_guessed():
    words, _ = parse_lyrics("hoje como mingau para comer")
    model = Model()
    report = retry_unresolved_gaps(model, np.zeros(16000 * 60),
                                   words, 0, 50, "pt")
    assert report["recovered"] == 0
    assert not model.calls
    assert all(w["start"] is None for w in words)


def test_gap_retry_preserves_prior_word_times():
    words, _ = parse_lyrics("hoje como mingau para")
    words[0].update(start=1.0, end=1.4)
    words[3].update(start=3.5, end=3.9)
    retry_unresolved_gaps(Model(), np.zeros(16000 * 6),
                          words, 0, 5, "pt")
    assert (words[0]["start"], words[0]["end"]) == (1.0, 1.4)
    assert (words[3]["start"], words[3]["end"]) == (3.5, 3.9)
