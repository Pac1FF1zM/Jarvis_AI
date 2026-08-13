"""Target-independent typed input features for direct Structured JSC."""
from __future__ import annotations

from modules.command_router import route_explicit_command, split_compound_command
from core.russian_numbers import extract_russian_cardinals

from .data import DialogueTurn, JSCExample
from .jal import JALPlan, dumps
from .sequence_data import normalize_utterance, serialize_source


def serialize_structured_source(example: JSCExample) -> str:
    """Add typed runtime evidence without consulting the training target."""
    return serialize_structured_input(example.text, example.history, example.state)


def serialize_structured_input(
    text: str,
    history: tuple[DialogueTurn, ...] = (),
    state: JALPlan | None = None,
) -> str:
    """Serialize a live turn using the exact training-time representation."""
    lines: list[str] = []
    for turn in history:
        prefix = f"H_{turn.role.upper()}"
        normalized_turn = normalize_utterance(turn.text)
        lines.append(f"{prefix}:{normalized_turn}")
        numbers = extract_russian_cardinals(normalized_turn)
        if numbers:
            lines.append(f"{prefix}_NUM:{','.join(map(str, numbers))}")
    if state is not None:
        lines.append(f"STATE:{dumps(state)}")
    normalized = normalize_utterance(text)
    lines.append(f"USER:{normalized}")
    numbers = extract_russian_cardinals(normalized)
    if numbers:
        lines.append(f"USER_NUM:{','.join(map(str, numbers))}")
    if state is not None:
        lines.append(f"STATE_ACT:{state.act.value}")
        for index, step in enumerate(state.steps):
            lines.append(f"STATE_TOOL_{index}:{step.tool}")
        for slot in state.missing:
            lines.append(f"STATE_MISSING_{slot.step}:{slot.name}")

    route_index = 0
    for part in split_compound_command(normalized):
        action = route_explicit_command(part)
        if action is None:
            continue
        lines.append(f"ROUTE_{route_index}:{action.intent}")
        route_index += 1
    return "\n".join(lines)


def structured_segment_targets(
    source_text: str,
    step_count: int,
    *,
    max_steps: int = 8,
) -> tuple[list[int], list[float], list[bool]]:
    """Build character-aligned action segments for boundary supervision.

    Token position zero is BOS, so every source character is shifted by one.
    Only the live ``USER:`` line participates; history, state, numeric and
    deterministic route hints remain available to attention but cannot become
    action boundaries.
    """
    if not 0 <= step_count <= max_steps:
        raise ValueError("step_count must fit structured max_steps")
    width = len(source_text) + 2
    segment_ids = [-1] * width
    boundaries = [0.0] * width
    boundary_mask = [False] * width
    marker = source_text.rfind("USER:")
    if marker < 0:
        raise ValueError("structured source has no USER line")
    user_start = marker + len("USER:")
    user_end = source_text.find("\n", user_start)
    if user_end < 0:
        user_end = len(source_text)
    # All real source characters are supervised so metadata cannot become a
    # high-scoring boundary merely because it was ignored by the loss.
    for position in range(1, len(source_text) + 1):
        boundary_mask[position] = True
    if step_count == 0 or user_start == user_end:
        return segment_ids, boundaries, boundary_mask

    utterance = source_text[user_start:user_end]
    parts = split_compound_command(utterance)
    relative_starts: list[int] = []
    cursor = 0
    for part in parts:
        found = _locate_compound_part(utterance, part, cursor)
        if found is None:
            continue
        relative_starts.append(found)
        cursor = max(found + 1, cursor)
    relative_starts = sorted(set(relative_starts))
    if not relative_starts or relative_starts[0] != 0:
        relative_starts.insert(0, 0)
    if len(relative_starts) > step_count:
        relative_starts = relative_starts[:step_count]
    while len(relative_starts) < step_count:
        # A coordinated application list may repeat a verb only implicitly.
        # Evenly spaced fallbacks still teach ordered, non-collapsed segments.
        candidate = round(len(utterance) * len(relative_starts) / step_count)
        candidate = min(max(candidate, relative_starts[-1] + 1), len(utterance) - 1)
        if candidate in relative_starts:
            break
        relative_starts.append(candidate)
    relative_starts = sorted(relative_starts[:step_count])

    for segment, relative_start in enumerate(relative_starts):
        absolute_start = user_start + relative_start + 1
        boundaries[absolute_start] = 1.0
        relative_end = (
            relative_starts[segment + 1]
            if segment + 1 < len(relative_starts)
            else len(utterance)
        )
        for relative_position in range(relative_start, relative_end):
            segment_ids[user_start + relative_position + 1] = segment
    return segment_ids, boundaries, boundary_mask


def _locate_compound_part(text: str, part: str, cursor: int) -> int | None:
    direct = text.find(part, cursor)
    if direct >= 0:
        return direct
    # Enumerations are expanded by the deterministic splitter (for example,
    # "закрой калькулятор и проводник" -> "закрой проводник"). Locate the
    # longest suffix that is actually present in the original utterance.
    words = part.split()
    for index in range(1, len(words)):
        suffix = " ".join(words[index:])
        found = text.find(suffix, cursor)
        if found >= 0:
            return found
    return None
