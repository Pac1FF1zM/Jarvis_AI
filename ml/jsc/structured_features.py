"""Target-independent typed input features for direct Structured JSC."""
from __future__ import annotations

from modules.command_router import route_explicit_command, split_compound_command

from .data import JSCExample
from .sequence_data import normalize_utterance, serialize_source


def serialize_structured_source(example: JSCExample) -> str:
    """Add typed runtime evidence without consulting the training target."""
    lines = [serialize_source(example)]
    if example.state is not None:
        lines.append(f"STATE_ACT:{example.state.act.value}")
        for index, step in enumerate(example.state.steps):
            lines.append(f"STATE_TOOL_{index}:{step.tool}")
        for slot in example.state.missing:
            lines.append(f"STATE_MISSING_{slot.step}:{slot.name}")

    normalized = normalize_utterance(example.text)
    route_index = 0
    for part in split_compound_command(normalized):
        action = route_explicit_command(part)
        if action is None:
            continue
        lines.append(f"ROUTE_{route_index}:{action.intent}")
        route_index += 1
    return "\n".join(lines)
