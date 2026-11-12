"""Second-pass word ASR recovery uses real observed timestamps only."""
import numpy as np
from music_lyrics_analyzer.core import parse_lyrics
from music_lyrics_analyzer.recovery import recover_with_word_asr


def transcript(text):
    return [{"text": t, "start": 1 + i * .5, "end": 1.35 + i * .5}
            for i, t in enumerate(text.split())]


def test_recovers_mistranscribed_inner_words_with_tentative_status():
    words, _ = parse_lyrics("Hoje como mingal para não comer arroz")
    result = recover_with_word_asr(
        None, np.zeros(16000), words, 0, 8, "pt",
        transcript=transcript("Hoje momo cagal para não comer suafoz"))
    assert result["recovered"] == 6
    assert words[1]["alignment"]["status"] == "tentative"
    assert words[2]["text"] == "mingal"
    assert words[2]["start"] == 2.0
    assert words[-1]["start"] is None


def test_does_not_overwrite_forced_alignment():
    words, _ = parse_lyrics("Hoje como mingal para")
    words[1].update(start=1.4, end=1.8)
    result = recover_with_word_asr(
        None, np.zeros(16000), words, 0, 5, "pt",
        transcript=transcript("Hoje momo cagal para"))
    assert words[1]["start"] == 1.4
    assert result["recovered"] == 3


def test_does_not_guess_missing_asr_word():
    words, _ = parse_lyrics("Hoje eu canto muito bem")
    recover_with_word_asr(
        None, np.zeros(16000), words, 0, 5, "pt",
        transcript=transcript("Hoje canto muito bem"))
    assert words[1]["start"] is None


def test_rejects_conflicting_boundaries():
    words, _ = parse_lyrics("Hoje como mingal para")
    words[0].update(start=1.0, end=2.2)
    recover_with_word_asr(
        None, np.zeros(16000), words, 0, 5, "pt",
        transcript=transcript("Hoje momo cagal para"))
    assert words[1]["start"] is None
    assert words[2]["start"] is None
    assert words[3]["start"] is not None


class MockWord:
    def __init__(self, text, start, end):
        self.word, self.start, self.end = text, start, end
        self.probability = .9


class MockSegment:
    def __init__(self, words):
        self.words = words


class MockModel:
    def transcribe(self, audio, **kwargs):
        assert kwargs["word_timestamps"] is True
        class Result:
            segments = [MockSegment([MockWord("Hoje", 0.1, .3),
                                     MockWord("momo", .4, .7),
                                     MockWord("para", .8, 1.1)])]
        return Result()


def test_transcription_rebased_to_absolute_window():
    from music_lyrics_analyzer.recovery import recognize_word_timestamps
    audio = np.zeros(16000 * 6, dtype=np.float32)
    items = recognize_word_timestamps(MockModel(), audio, 2.0, 5.0, "pt")
    assert [x["start"] for x in items] == [2.1, 2.4, 2.8]


def test_no_word_timestamp_no_recovery():
    words, _ = parse_lyrics("Hoje tudo bem")
    recover_with_word_asr(
        None, np.zeros(16000), words, 0, 5, "pt",
        transcript=[{"text": "Hoje tudo bem", "start": 1, "end": 3}])
    assert all(w["start"] is None for w in words)
