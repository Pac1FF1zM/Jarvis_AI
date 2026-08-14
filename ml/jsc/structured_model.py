"""Non-autoregressive neural architecture for Structured JSC."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import nn


@dataclass(frozen=True)
class StructuredJSCConfig:
    vocab_size: int
    num_acts: int
    num_tools: int
    num_parameter_labels: int
    num_span_slots: int
    num_missing_labels: int
    num_reasons: int
    d_model: int = 192
    encoder_layers: int = 3
    step_layers: int = 2
    attention_heads: int = 4
    feedforward_dim: int = 384
    dropout: float = 0.12
    max_source_length: int = 384
    max_steps: int = 8
    pad_id: int = 0
    segmented_router: bool = False

    def __post_init__(self) -> None:
        dimensions = (
            self.vocab_size,
            self.num_acts,
            self.num_tools,
            self.num_parameter_labels,
            self.num_span_slots,
            self.num_missing_labels,
            self.num_reasons,
        )
        if any(value < 1 for value in dimensions):
            raise ValueError("structured label spaces must be non-empty")
        if self.d_model < 16 or self.d_model % self.attention_heads:
            raise ValueError("d_model must be >=16 and divisible by attention_heads")
        if self.encoder_layers < 1 or self.step_layers < 1 or self.max_steps < 1:
            raise ValueError("encoder_layers, step_layers and max_steps must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "StructuredJSCConfig":
        return cls(**dict(raw))


class StructuredJSCModel(nn.Module):
    """Encode an utterance once and directly predict a typed JAL program."""

    def __init__(self, config: StructuredJSCConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_id
        )
        self.source_positions = nn.Embedding(config.max_source_length, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.encoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )
        self.semantic_attention = nn.Linear(config.d_model, 1, bias=False)
        self.semantic_projection = nn.Sequential(
            nn.LayerNorm(config.d_model * 3),
            nn.Linear(config.d_model * 3, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.d_model),
        )
        self.step_embeddings = nn.Embedding(config.max_steps, config.d_model)
        step_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # This is a fixed-width parallel program reasoner, not a token or JSON
        # decoder.  Self-attention lets ordered step queries coordinate instead
        # of collapsing onto the same command fragment.
        self.step_reasoner = nn.TransformerDecoder(
            step_layer,
            num_layers=config.step_layers,
            norm=nn.LayerNorm(config.d_model),
        )
        self.tool_embeddings = nn.Embedding(config.num_tools, config.d_model)
        self.condition_projection = nn.Sequential(
            nn.LayerNorm(config.d_model * 2),
            nn.Linear(config.d_model * 2, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(config.d_model),
        )
        self.act_head = _classification_head(config, config.num_acts)
        self.reason_head = _classification_head(config, config.num_reasons)
        self.step_count_head = _classification_head(config, config.max_steps + 1)
        self.execution_verifier_head = _classification_head(config, 2)
        if config.segmented_router:
            self.boundary_head = nn.Linear(config.d_model, 1)
            self.segment_projection = nn.Sequential(
                nn.LayerNorm(config.d_model * 2),
                nn.Linear(config.d_model * 2, config.d_model),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.LayerNorm(config.d_model),
            )
            # Tool selection is intentionally isolated from argument/span
            # heads: one pooled command segment owns one ordered tool choice.
            self.segment_tool_head = nn.Linear(config.d_model, config.num_tools)
        else:
            self.tool_head = nn.Linear(config.d_model, config.num_tools)
        self.parameter_head = nn.Linear(config.d_model, config.num_parameter_labels)
        self.missing_head = nn.Linear(config.d_model, config.num_missing_labels)
        self.span_slot_embeddings = nn.Embedding(config.num_span_slots, config.d_model)
        self.span_start_query = nn.Linear(config.d_model, config.d_model)
        self.span_end_query = nn.Linear(config.d_model, config.d_model)
        self.span_memory = nn.Linear(config.d_model, config.d_model)
        self._reset_parameters()

    def forward(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        *,
        conditioning_tool_ids: torch.Tensor | None = None,
        conditioning_segment_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        memory, pooled = self.encode(source_ids, source_mask)
        step_states = self._step_states(memory, pooled, source_mask)
        act_logits = self.act_head(pooled)
        count_logits = self.step_count_head(pooled)
        boundary_logits: torch.Tensor | None = None
        if self.config.segmented_router:
            boundary_logits = self.boundary_head(memory).squeeze(-1)
            boundary_logits = boundary_logits.masked_fill(
                ~source_mask.bool(), torch.finfo(boundary_logits.dtype).min
            )
            segment_ids = (
                self._predicted_segment_ids(
                    boundary_logits, count_logits, source_mask
                )
                if conditioning_segment_ids is None
                else conditioning_segment_ids
            )
            segment_states = self._segment_states(
                memory, step_states, source_mask, segment_ids
            )
            routed_states = self.segment_projection(
                torch.cat((step_states, segment_states), dim=-1)
            )
            tool_logits = self.segment_tool_head(routed_states)
        else:
            tool_logits = self.tool_head(step_states)
        tool_ids = (
            tool_logits.argmax(dim=-1)
            if conditioning_tool_ids is None
            else conditioning_tool_ids
        )
        conditioned = self.condition_projection(
            torch.cat((step_states, self.tool_embeddings(tool_ids)), dim=-1)
        )
        starts, ends = self._span_scores(memory, conditioned, source_mask)
        outputs = (
            act_logits,
            count_logits,
            tool_logits,
            self.parameter_head(conditioned),
            starts,
            ends,
            self.execution_verifier_head(pooled),
            self.missing_head(conditioned),
            self.reason_head(pooled),
        )
        return outputs if boundary_logits is None else (*outputs, boundary_logits)

    def encode(
        self, source_ids: torch.Tensor, source_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if source_ids.shape[1] > self.config.max_source_length:
            raise ValueError("source exceeds configured positions")
        positions = torch.arange(source_ids.shape[1], device=source_ids.device)
        embedded = self.embedding_dropout(
            self.token_embedding(source_ids) + self.source_positions(positions)[None]
        )
        memory = self.encoder(embedded, src_key_padding_mask=~source_mask.bool())
        float_mask = source_mask.unsqueeze(-1).to(memory.dtype)
        mean = (memory * float_mask).sum(1) / float_mask.sum(1).clamp_min(1.0)
        logits = self.semantic_attention(memory).squeeze(-1)
        logits = logits.masked_fill(
            ~source_mask.bool(), torch.finfo(logits.dtype).min
        )
        attended = torch.einsum("bl,bld->bd", logits.softmax(-1), memory)
        maximum = memory.masked_fill(
            ~source_mask.unsqueeze(-1).bool(), torch.finfo(memory.dtype).min
        ).max(1).values
        pooled = self.semantic_projection(torch.cat((mean, attended, maximum), -1))
        return memory, pooled

    def _step_states(
        self,
        memory: torch.Tensor,
        pooled: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        queries = pooled[:, None, :] + self.step_embeddings.weight[None]
        return self.step_reasoner(
            queries,
            memory,
            memory_key_padding_mask=~source_mask.bool(),
        )

    def _span_scores(
        self,
        memory: torch.Tensor,
        conditioned_steps: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        queries = (
            conditioned_steps[:, :, None, :]
            + self.span_slot_embeddings.weight[None, None, :, :]
        )
        keys = self.span_memory(memory)
        scale = self.config.d_model**0.5
        starts = torch.einsum(
            "bksd,bld->bksl", self.span_start_query(queries), keys
        ) / scale
        ends = torch.einsum(
            "bksd,bld->bksl", self.span_end_query(queries), keys
        ) / scale
        invalid = ~source_mask[:, None, None, :].bool()
        minimum = torch.finfo(starts.dtype).min
        return starts.masked_fill(invalid, minimum), ends.masked_fill(invalid, minimum)

    def _segment_states(
        self,
        memory: torch.Tensor,
        step_states: torch.Tensor,
        source_mask: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> torch.Tensor:
        if segment_ids.shape != source_mask.shape:
            raise ValueError("conditioning_segment_ids must match source shape")
        states = []
        for index in range(self.config.max_steps):
            selected = segment_ids.eq(index) & source_mask.bool()
            weights = selected.unsqueeze(-1).to(memory.dtype)
            count = weights.sum(1)
            pooled = (memory * weights).sum(1) / count.clamp_min(1.0)
            pooled = torch.where(count.gt(0), pooled, step_states[:, index])
            states.append(pooled)
        return torch.stack(states, dim=1)

    def _predicted_segment_ids(
        self,
        boundary_logits: torch.Tensor,
        count_logits: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        result = torch.full_like(source_mask, -1, dtype=torch.long)
        counts = count_logits.argmax(-1).clamp(0, self.config.max_steps)
        for row in range(boundary_logits.shape[0]):
            valid = source_mask[row].bool().clone()
            valid[0] = False
            valid_indices = valid.nonzero(as_tuple=False).flatten()
            if valid_indices.numel():
                valid[valid_indices[-1]] = False  # EOS
            count = min(int(counts[row]), int(valid.sum()))
            if count < 1:
                continue
            scores = boundary_logits[row].masked_fill(
                ~valid, torch.finfo(boundary_logits.dtype).min
            )
            starts = scores.topk(count).indices.sort().values
            final = int(valid_indices[-1]) if valid_indices.numel() else source_mask.shape[1]
            for segment, start in enumerate(starts.tolist()):
                end = int(starts[segment + 1]) if segment + 1 < count else final
                result[row, start:end] = segment
        return result

    def _reset_parameters(self) -> None:
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        with torch.no_grad():
            self.token_embedding.weight[self.config.pad_id].zero_()

    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters())


def _classification_head(
    config: StructuredJSCConfig, outputs: int
) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(config.d_model),
        nn.Dropout(config.dropout),
        nn.Linear(config.d_model, outputs),
    )
