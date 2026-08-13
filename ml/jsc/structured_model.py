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
    attention_heads: int = 4
    feedforward_dim: int = 384
    dropout: float = 0.12
    max_source_length: int = 384
    max_steps: int = 8
    pad_id: int = 0

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
        if self.encoder_layers < 1 or self.max_steps < 1:
            raise ValueError("encoder_layers and max_steps must be positive")
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
        self.step_attention = nn.MultiheadAttention(
            config.d_model,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.step_norm = nn.LayerNorm(config.d_model)
        self.step_feedforward = nn.Sequential(
            nn.Linear(config.d_model, config.feedforward_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward_dim, config.d_model),
            nn.Dropout(config.dropout),
        )
        self.step_output_norm = nn.LayerNorm(config.d_model)
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
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        memory, pooled = self.encode(source_ids, source_mask)
        step_states = self._step_states(memory, pooled, source_mask)
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
        return (
            self.act_head(pooled),
            self.step_count_head(pooled),
            tool_logits,
            self.parameter_head(conditioned),
            starts,
            ends,
            self.execution_verifier_head(pooled),
            self.missing_head(conditioned),
            self.reason_head(pooled),
        )

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
        logits = self.semantic_attention(memory).squeeze(-1).masked_fill(
            ~source_mask.bool(), torch.finfo(memory.dtype).min
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
        attended, _weights = self.step_attention(
            queries,
            memory,
            memory,
            key_padding_mask=~source_mask.bool(),
            need_weights=False,
        )
        states = self.step_norm(queries + attended)
        return self.step_output_norm(states + self.step_feedforward(states))

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
