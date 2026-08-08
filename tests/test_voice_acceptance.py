from __future__ import annotations

from pathlib import Path

from core.voice_acceptance import evaluate_candidates, parse_session_logs, summarise_turns


def test_parser_keeps_only_anonymised_complete_real_turns(tmp_path: Path):
    log = tmp_path / "session.txt"
    log.write_text(
        "\n".join(
            [
                "2026-08-09 10:00:00.000 INFO jarvis.wake | VOICE_ACTIVATED source=wake_phrase hotkey=x",
                "2026-08-09 10:00:00.001 INFO jarvis.bus | PUBLISH wake_word_detected trace=abc123",
                "2026-08-09 10:00:01.001 INFO jarvis.bus | PUBLISH wake_acknowledgement_finished trace=abc123",
                "2026-08-09 10:00:01.201 INFO jarvis.bus | PUBLISH speech_capture_started trace=abc123",
                "2026-08-09 10:00:03.201 INFO jarvis.wake | AUDIO_CAPTURED real trace=abc123 bytes=1 duration_ms=2200 end=vad_silence",
                "2026-08-09 10:00:04.201 INFO jarvis.stt | TRANSCRIPTION_READY trace=abc123 text='private words' conf=.9",
                "2026-08-09 10:00:04.221 INFO jarvis.nlu | NLU_RESULT trace=abc123 intent=x slots={}",
                "2026-08-09 10:00:04.321 INFO jarvis.bus | PUBLISH response_ready trace=abc123",
                "2026-08-09 10:00:05.321 INFO jarvis.tts | SPEECH_FINISHED trace=abc123 generation=1",
                "2026-08-09 10:00:06.000 INFO jarvis.bus | PUBLISH wake_word_detected trace=incomplete",
            ]
        ),
        encoding="utf-8",
    )

    turns = parse_session_logs([log])

    assert len(turns) == 1
    assert turns[0].source == "wake_phrase"
    assert turns[0].acknowledgement_ms == 1000
    assert turns[0].capture_ms == 2200
    assert turns[0].stt_ms == 1000
    assert not hasattr(turns[0], "transcript")


def test_balanced_candidate_wins_without_breaking_voice_requirements(tmp_path: Path):
    log = tmp_path / "session.txt"
    lines = []
    for index, capture in enumerate((2200, 2500, 15000)):
        minute = index * 2
        trace = f"abc{index}"
        lines.extend(
            [
                f"2026-08-09 10:{minute:02d}:00.000 INFO wake | VOICE_ACTIVATED source=wake_phrase hotkey=x",
                f"2026-08-09 10:{minute:02d}:00.001 INFO bus | PUBLISH wake_word_detected trace={trace}",
                f"2026-08-09 10:{minute:02d}:02.401 INFO bus | PUBLISH wake_acknowledgement_finished trace={trace}",
                f"2026-08-09 10:{minute:02d}:02.801 INFO bus | PUBLISH speech_capture_started trace={trace}",
                f"2026-08-09 10:{minute:02d}:17.801 INFO wake | AUDIO_CAPTURED real trace={trace} duration_ms={capture} end={'max_duration' if capture == 15000 else 'vad_silence'}",
                f"2026-08-09 10:{minute:02d}:21.801 INFO stt | TRANSCRIPTION_READY trace={trace} text='x'",
                f"2026-08-09 10:{minute:02d}:21.821 INFO nlu | NLU_RESULT trace={trace} intent=x",
                f"2026-08-09 10:{minute:02d}:21.921 INFO bus | PUBLISH response_ready trace={trace}",
                f"2026-08-09 10:{minute:02d}:23.421 INFO tts | SPEECH_FINISHED trace={trace}",
            ]
        )
    log.write_text("\n".join(lines), encoding="utf-8")

    turns = parse_session_logs([log])
    results = evaluate_candidates(turns)

    assert results[0].candidate.name == "balanced_vad"
    assert results[0].eligible is True
    assert next(item for item in results if item.candidate.name == "short_ack").eligible is False
    assert next(item for item in results if item.candidate.name == "duplex_overlap").eligible is False
    assert summarise_turns(turns)["max_duration_captures"] == 1
