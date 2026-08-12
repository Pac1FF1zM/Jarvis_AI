"""Fail-closed gate between a final STT transcript and semantic actions."""
from __future__ import annotations

import re
from dataclasses import dataclass


_IMPERATIVE_VERB = (
    r"открой|запусти|включи|выключи|закрой|сверни|разверни|найди|поищи|"
    r"переключи|создай|покажи|напомни"
)
_COMMAND_VERB = rf"{_IMPERATIVE_VERB}|открыть|запустить"
_META_SPEECH = re.compile(
    rf"^(?:(?:он|она|они|кто-то)\s+(?:сказал|сказала|сказали|говорит|произнес)|"
    rf"(?:что|а что)\s+(?:будет|произойдет),?\s+если\s+(?:сказать|произнести)|"
    rf"(?:команда|фраза)\s+[«\"']?.*\b(?:{_COMMAND_VERB})\b)",
    flags=re.IGNORECASE,
)
_TRAILING_INCOMPLETE = re.compile(
    rf"(?:\b(?:{_IMPERATIVE_VERB})|\b(?:и|а|но|затем|потом))\s*[.!?…]*$",
    flags=re.IGNORECASE,
)
_LEADING_COMMAND = re.compile(
    rf"^\s*(?:(?:джарвис|пожалуйста)\s+)*(?P<verb>{_IMPERATIVE_VERB})\b",
    flags=re.IGNORECASE,
)
_CORRECTION = re.compile(
    r"^(?P<before>.+?)(?:[.!?…]+\s*|\s+)(?P<marker>ой|нет|стоп|подожди)\s*[,.:;-]*\s*(?P<after>.+)$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class GateResult:
    state: str
    reason: str
    route_text: str


def prepare_final_utterance(text: str) -> GateResult:
    """Decide whether a completed capture may reach semantic parsing."""
    value = re.sub(r"\s+", " ", text.casefold().replace("ё", "е")).strip()
    if not value:
        return GateResult("rejected", "empty_transcript", "")
    if _META_SPEECH.search(value):
        return GateResult("rejected", "mentioned_or_quoted_command", "")
    if re.match(
        r"^не\s+(?:открывай|запускай|включай|выключай|закрывай|ищи|переключай|создавай)\b",
        value,
    ):
        return GateResult("rejected", "negated_command", "")

    correction = _CORRECTION.match(value)
    if correction:
        before = correction.group("before").strip(" ,.:;!?…-")
        after = correction.group("after").strip(" ,.:;!?…-")
        if re.fullmatch(r"(?:не надо|отмена|не выполняй|ничего)", after):
            return GateResult("rejected", "self_cancelled", "")
        if _LEADING_COMMAND.match(after):
            value = after
        else:
            better = re.match(r"^(?:лучше|вместо этого)\s+(.+)$", after)
            original = _LEADING_COMMAND.match(before)
            if better and original:
                value = f"{original.group('verb')} {better.group(1)}"
            else:
                return GateResult("clarify", "ambiguous_self_correction", "")

    if _TRAILING_INCOMPLETE.search(value):
        return GateResult("wait", "unfinished_utterance", "")
    return GateResult("analyze", "final_utterance", value)
