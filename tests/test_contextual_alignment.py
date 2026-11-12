"""Tests for transcript anchors without any speech-model download."""
from music_lyrics_analyzer.contextual_alignment import align_transcript_words


def words(text):
    return [{"id": str(i), "text": token} for i, token in enumerate(text.split())]


def transcript(text):
    return [{"text": token, "start": i * .4, "end": i * .4 + .3}
            for i, token in enumerate(text.split())]


def test_noisy_example_preserves_known_lyrics():
    reference = words("Hoje como mingal para não comer arroz")
    heard = transcript("Hoje momo cagal para não comer suafoz")
    result = align_transcript_words(reference, heard)
    assert [r["status"] for r in result] == [
        "exact", "contextual", "contextual", "exact", "exact", "exact", "unresolved"]
    assert result[1]["start"] == .4
    assert result[2]["text"] == "mingal"
    assert result[-1]["start"] is None


def test_no_invented_timing_when_asr_omits_word():
    result = align_transcript_words(words("hoje eu canto muito bem"),
                                    transcript("hoje canto muito bem"))
    assert result[1]["status"] == "unresolved"
    assert result[1]["start"] is None


def test_repeated_words_use_distinct_occurrences():
    result = align_transcript_words(words("foi só uma vez foi só uma vez"),
                                    transcript("foi só uma vez foi só uma vez"))
    assert all(r["status"] == "exact" for r in result)
    assert len(set(r["asr_index"] for r in result)) == 8


def test_no_context_recovery_when_counts_disagree():
    result = align_transcript_words(words("hoje eu tomo mingau para comer"),
                                    transcript("hoje zzz para comer"))
    assert all(r["start"] is None for r in result[1:4])


def test_invalid_times_not_used():
    heard = transcript("hoje tudo bem")
    heard[1]["start"] = None
    result = align_transcript_words(words("hoje tudo bem"), heard)
    assert result[1]["start"] is None
