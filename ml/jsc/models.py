"""From-scratch sequence baselines for Jarvis Semantic Core."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import nn


ARCHITECTURES = ("char_cnn", "bigru", "tiny_transformer")


@dataclass(frozen=True)
class BaselineConfig:
    architecture: str
    vocab_size: int
    num_acts: int
    d_model: int = 128
    encoder_layers: int = 2
    decoder_layers: int = 2
    attention_heads: int = 4
    feedforward_dim: int = 256
    dropout: float = 0.15
    max_source_length: int = 384
    max_target_length: int = 384
    pad_id: int = 0
    copy_mechanism: bool = False
    num_tools: int = 0
    num_parameter_labels: int = 0
    num_span_slots: int = 0
    semantic_pooling: bool = False
    execution_verifier: bool = False
    max_steps: int = 8

    def __post_init__(self) -> None:
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"unknown JSC baseline architecture {self.architecture!r}")
        if self.vocab_size < 8 or self.num_acts < 2:
            raise ValueError("model label spaces are implausibly small")
        if self.d_model < 16 or self.d_model % 2:
            raise ValueError("d_model must be even and at least 16")
        if self.d_model % self.attention_heads:
            raise ValueError("d_model must be divisible by attention_heads")
        if self.encoder_layers < 1 or self.decoder_layers < 1:
            raise ValueError("encoder and decoder need at least one layer")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if (
            self.num_tools < 0
            or self.num_parameter_labels < 0
            or self.num_span_slots < 0
            or self.max_steps < 1
        ):
            raise ValueError("structured label dimensions are invalid")
        if self.num_parameter_labels and not self.num_tools:
            raise ValueError("parameter heads require structured tool heads")
        if self.num_span_slots and not self.num_tools:
            raise ValueError("span heads require structured tool heads")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> "BaselineConfig":
        return cls(**{key: value for key, value in raw.items()})


class JSCBaselineModel(nn.Module):
    """A shared autoregressive decoder over three independently useful encoders."""

    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_id,
        )
        self.source_positions = nn.Embedding(config.max_source_length, config.d_model)
        self.target_positions = nn.Embedding(config.max_target_length, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        if config.architecture == "char_cnn":
            self.encoder = _CNNEncoder(config)
        elif config.architecture == "bigru":
            self.encoder = _BiGRUEncoder(config)
        else:
            self.encoder = _TransformerEncoder(config)
        if config.semantic_pooling:
            self.semantic_attention = nn.Linear(config.d_model, 1, bias=False)
            self.semantic_projection = nn.Sequential(
                nn.LayerNorm(config.d_model * 3),
                nn.Linear(config.d_model * 3, config.d_model),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.LayerNorm(config.d_model),
            )
        else:
            self.semantic_attention = None
            self.semantic_projection = None
        self.execution_verifier_head = (
            nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model, 2),
            )
            if config.execution_verifier
            else None
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.decoder_layers,
            norm=nn.LayerNorm(config.d_model),
        )
        self.token_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.token_head.weight = self.token_embedding.weight
        if config.copy_mechanism:
            self.copy_query = nn.Linear(config.d_model, config.d_model, bias=False)
            self.copy_gate = nn.Linear(config.d_model * 2, 1)
        else:
            self.copy_query = None
            self.copy_gate = None
        self.act_head = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.num_acts),
        )
        if config.num_tools:
            self.step_embeddings = nn.Embedding(config.max_steps, config.d_model)
            self.step_count_head = nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model, config.max_steps + 1),
            )
            self.tool_sequence_head = nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model, config.num_tools),
            )
            self.parameter_head = (
                nn.Sequential(
                    nn.LayerNorm(config.d_model),
                    nn.Dropout(config.dropout),
                    nn.Linear(config.d_model, config.num_parameter_labels),
                )
                if config.num_parameter_labels
                else None
            )
            if config.num_span_slots:
                self.span_slot_embeddings = nn.Embedding(
                    config.num_span_slots, config.d_model
                )
                self.span_start_query = nn.Linear(config.d_model, config.d_model)
                self.span_end_query = nn.Linear(config.d_model, config.d_model)
                self.span_memory = nn.Linear(config.d_model, config.d_model)
            else:
                self.span_slot_embeddings = None
                self.span_start_query = None
                self.span_end_query = None
                self.span_memory = None
        else:
            self.step_embeddings = None
            self.step_count_head = None
            self.tool_sequence_head = None
            self.parameter_head = None
            self.span_slot_embeddings = None
            self.span_start_query = None
            self.span_end_query = None
            self.span_memory = None
        self._reset_parameters()

    def forward(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        memory, pooled = self.encode(source_ids, source_mask)
        semantic_pooled = self._semantic_pooled(memory, pooled, source_mask)
        decoded = self._decode(decoder_input_ids, decoder_mask, memory, source_mask)
        return (
            self._token_scores(decoded, memory, source_ids, source_mask),
            self.act_head(semantic_pooled),
        )

    def forward_structured(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.step_count_head is None or self.tool_sequence_head is None:
            raise RuntimeError("structured heads are disabled")
        memory, pooled = self.encode(source_ids, source_mask)
        semantic_pooled = self._semantic_pooled(memory, pooled, source_mask)
        decoded = self._decode(decoder_input_ids, decoder_mask, memory, source_mask)
        count_logits, tool_logits = self._structured_scores(semantic_pooled)
        return (
            self._token_scores(decoded, memory, source_ids, source_mask),
            self.act_head(semantic_pooled),
            count_logits,
            tool_logits,
        )

    def forward_schema_conditioned(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.parameter_head is None:
            raise RuntimeError("schema-conditioned parameter head is disabled")
        memory, pooled = self.encode(source_ids, source_mask)
        semantic_pooled = self._semantic_pooled(memory, pooled, source_mask)
        decoded = self._decode(decoder_input_ids, decoder_mask, memory, source_mask)
        count_logits, tool_logits, parameter_logits = self._schema_scores(
            semantic_pooled, pooled
        )
        return (
            self._token_scores(decoded, memory, source_ids, source_mask),
            self.act_head(semantic_pooled),
            count_logits,
            tool_logits,
            parameter_logits,
        )

    def forward_full_semantic(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if self.parameter_head is None or self.span_slot_embeddings is None:
            raise RuntimeError("full semantic heads are disabled")
        memory, pooled = self.encode(source_ids, source_mask)
        semantic_pooled = self._semantic_pooled(memory, pooled, source_mask)
        decoded = self._decode(decoder_input_ids, decoder_mask, memory, source_mask)
        count_logits, tool_logits, parameter_logits = self._schema_scores(
            semantic_pooled, pooled
        )
        start_logits, end_logits = self._span_scores(memory, pooled, source_mask)
        return (
            self._token_scores(decoded, memory, source_ids, source_mask),
            self.act_head(semantic_pooled),
            count_logits,
            tool_logits,
            parameter_logits,
            start_logits,
            end_logits,
        )

    def forward_verified_semantic(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        decoder_mask: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if self.execution_verifier_head is None:
            raise RuntimeError("execution verifier head is disabled")
        if self.parameter_head is None or self.span_slot_embeddings is None:
            raise RuntimeError("full semantic heads are disabled")
        memory, pooled = self.encode(source_ids, source_mask)
        semantic_pooled = self._semantic_pooled(memory, pooled, source_mask)
        decoded = self._decode(decoder_input_ids, decoder_mask, memory, source_mask)
        count_logits, tool_logits, parameter_logits = self._schema_scores(
            semantic_pooled, pooled
        )
        start_logits, end_logits = self._span_scores(memory, pooled, source_mask)
        return (
            self._token_scores(decoded, memory, source_ids, source_mask),
            self.act_head(semantic_pooled),
            count_logits,
            tool_logits,
            parameter_logits,
            start_logits,
            end_logits,
            self.execution_verifier_head(semantic_pooled),
        )

    def encode(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embedded = self._embed(
            source_ids,
            self.source_positions,
            self.config.max_source_length,
        )
        memory = self.encoder(embedded, source_mask)
        float_mask = source_mask.unsqueeze(-1).to(memory.dtype)
        pooled = (memory * float_mask).sum(dim=1) / float_mask.sum(dim=1).clamp_min(1.0)
        return memory, pooled

    def _semantic_pooled(
        self,
        memory: torch.Tensor,
        mean_pooled: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.semantic_attention is None or self.semantic_projection is None:
            return mean_pooled
        attention_logits = self.semantic_attention(memory).squeeze(-1)
        attention_logits = attention_logits.masked_fill(
            ~source_mask.bool(), torch.finfo(attention_logits.dtype).min
        )
        attended = torch.einsum(
            "bl,bld->bd", attention_logits.softmax(dim=-1), memory
        )
        masked_memory = memory.masked_fill(
            ~source_mask.unsqueeze(-1).bool(), torch.finfo(memory.dtype).min
        )
        maximum = masked_memory.max(dim=1).values
        return self.semantic_projection(
            torch.cat((mean_pooled, attended, maximum), dim=-1)
        )

    @torch.inference_mode()
    def greedy_decode(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        *,
        bos_id: int,
        eos_id: int,
        max_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.eval()
        limit = max_length or self.config.max_target_length
        if limit > self.config.max_target_length:
            raise ValueError("decode limit exceeds configured target positions")
        memory, pooled = self.encode(source_ids, source_mask)
        semantic_pooled = self._semantic_pooled(memory, pooled, source_mask)
        generated = torch.full(
            (source_ids.shape[0], 1),
            bos_id,
            dtype=torch.long,
            device=source_ids.device,
        )
        finished = torch.zeros(source_ids.shape[0], dtype=torch.bool, device=source_ids.device)
        for _ in range(limit - 1):
            decoder_mask = generated.ne(self.config.pad_id)
            decoded = self._decode(generated, decoder_mask, memory, source_mask)
            next_ids = self._token_scores(
                decoded[:, -1:], memory, source_ids, source_mask
            )[:, -1].argmax(dim=-1)
            next_ids = torch.where(finished, torch.full_like(next_ids, eos_id), next_ids)
            generated = torch.cat((generated, next_ids.unsqueeze(1)), dim=1)
            finished |= next_ids.eq(eos_id)
            if bool(finished.all()):
                break
        return generated, self.act_head(semantic_pooled)

    @torch.inference_mode()
    def greedy_decode_structured(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        *,
        bos_id: int,
        eos_id: int,
        max_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.step_count_head is None or self.tool_sequence_head is None:
            raise RuntimeError("structured heads are disabled")
        self.eval()
        limit = max_length or self.config.max_target_length
        if limit > self.config.max_target_length:
            raise ValueError("decode limit exceeds configured target positions")
        memory, pooled = self.encode(source_ids, source_mask)
        semantic_pooled = self._semantic_pooled(memory, pooled, source_mask)
        generated = torch.full(
            (source_ids.shape[0], 1), bos_id, dtype=torch.long, device=source_ids.device
        )
        finished = torch.zeros(source_ids.shape[0], dtype=torch.bool, device=source_ids.device)
        for _ in range(limit - 1):
            decoder_mask = generated.ne(self.config.pad_id)
            decoded = self._decode(generated, decoder_mask, memory, source_mask)
            next_ids = self._token_scores(
                decoded[:, -1:], memory, source_ids, source_mask
            )[:, -1].argmax(dim=-1)
            next_ids = torch.where(finished, torch.full_like(next_ids, eos_id), next_ids)
            generated = torch.cat((generated, next_ids.unsqueeze(1)), dim=1)
            finished |= next_ids.eq(eos_id)
            if bool(finished.all()):
                break
        count_logits, tool_logits = self._structured_scores(semantic_pooled)
        return generated, self.act_head(semantic_pooled), count_logits, tool_logits

    @torch.inference_mode()
    def greedy_decode_schema_conditioned(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        *,
        bos_id: int,
        eos_id: int,
        max_length: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.parameter_head is None:
            raise RuntimeError("schema-conditioned parameter head is disabled")
        self.eval()
        limit = max_length or self.config.max_target_length
        if limit > self.config.max_target_length:
            raise ValueError("decode limit exceeds configured target positions")
        memory, pooled = self.encode(source_ids, source_mask)
        semantic_pooled = self._semantic_pooled(memory, pooled, source_mask)
        generated = torch.full(
            (source_ids.shape[0], 1), bos_id, dtype=torch.long, device=source_ids.device
        )
        finished = torch.zeros(source_ids.shape[0], dtype=torch.bool, device=source_ids.device)
        for _ in range(limit - 1):
            decoder_mask = generated.ne(self.config.pad_id)
            decoded = self._decode(generated, decoder_mask, memory, source_mask)
            next_ids = self._token_scores(
                decoded[:, -1:], memory, source_ids, source_mask
            )[:, -1].argmax(dim=-1)
            next_ids = torch.where(finished, torch.full_like(next_ids, eos_id), next_ids)
            generated = torch.cat((generated, next_ids.unsqueeze(1)), dim=1)
            finished |= next_ids.eq(eos_id)
            if bool(finished.all()):
                break
        count_logits, tool_logits, parameter_logits = self._schema_scores(
            semantic_pooled, pooled
        )
        return (
            generated,
            self.act_head(semantic_pooled),
            count_logits,
            tool_logits,
            parameter_logits,
        )

    @torch.inference_mode()
    def greedy_decode_full_semantic(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        *,
        bos_id: int,
        eos_id: int,
        max_length: int | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if self.parameter_head is None or self.span_slot_embeddings is None:
            raise RuntimeError("full semantic heads are disabled")
        self.eval()
        generated, act_logits, count_logits, tool_logits, parameter_logits = (
            self.greedy_decode_schema_conditioned(
                source_ids,
                source_mask,
                bos_id=bos_id,
                eos_id=eos_id,
                max_length=max_length,
            )
        )
        memory, pooled = self.encode(source_ids, source_mask)
        start_logits, end_logits = self._span_scores(memory, pooled, source_mask)
        return (
            generated,
            act_logits,
            count_logits,
            tool_logits,
            parameter_logits,
            start_logits,
            end_logits,
        )

    @torch.inference_mode()
    def greedy_decode_verified_semantic(
        self,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
        *,
        bos_id: int,
        eos_id: int,
        max_length: int | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if self.execution_verifier_head is None:
            raise RuntimeError("execution verifier head is disabled")
        outputs = self.greedy_decode_full_semantic(
            source_ids,
            source_mask,
            bos_id=bos_id,
            eos_id=eos_id,
            max_length=max_length,
        )
        memory, pooled = self.encode(source_ids, source_mask)
        semantic_pooled = self._semantic_pooled(memory, pooled, source_mask)
        return (*outputs, self.execution_verifier_head(semantic_pooled))

    def _structured_scores(self, pooled: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.step_embeddings is not None
        assert self.step_count_head is not None and self.tool_sequence_head is not None
        positions = self.step_embeddings.weight.unsqueeze(0)
        step_states = pooled.unsqueeze(1) + positions
        return self.step_count_head(pooled), self.tool_sequence_head(step_states)

    def _schema_scores(
        self,
        structured_pooled: torch.Tensor,
        parameter_pooled: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.parameter_head is None:
            raise RuntimeError("schema-conditioned parameter head is disabled")
        count_logits, tool_logits = self._structured_scores(structured_pooled)
        assert self.step_embeddings is not None
        step_states = (
            parameter_pooled.unsqueeze(1) + self.step_embeddings.weight.unsqueeze(0)
        )
        return count_logits, tool_logits, self.parameter_head(step_states)

    def _span_scores(
        self,
        memory: torch.Tensor,
        pooled: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self.span_slot_embeddings is None
            or self.span_start_query is None
            or self.span_end_query is None
            or self.span_memory is None
        ):
            raise RuntimeError("span heads are disabled")
        assert self.step_embeddings is not None
        queries = (
            pooled[:, None, None, :]
            + self.step_embeddings.weight[None, :, None, :]
            + self.span_slot_embeddings.weight[None, None, :, :]
        )
        keys = self.span_memory(memory)
        scale = self.config.d_model**0.5
        start_logits = torch.einsum(
            "bksd,bld->bksl", self.span_start_query(queries), keys
        ) / scale
        end_logits = torch.einsum(
            "bksd,bld->bksl", self.span_end_query(queries), keys
        ) / scale
        invalid = ~source_mask[:, None, None, :].bool()
        minimum = torch.finfo(start_logits.dtype).min
        return (
            start_logits.masked_fill(invalid, minimum),
            end_logits.masked_fill(invalid, minimum),
        )

    @property
    def token_scores_are_log_probabilities(self) -> bool:
        return self.config.copy_mechanism

    def _token_scores(
        self,
        decoded: torch.Tensor,
        memory: torch.Tensor,
        source_ids: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        vocabulary_logits = self.token_head(decoded)
        if not self.config.copy_mechanism:
            return vocabulary_logits
        assert self.copy_query is not None and self.copy_gate is not None
        attention_scores = torch.matmul(
            self.copy_query(decoded), memory.transpose(1, 2)
        ) / (self.config.d_model**0.5)
        # IDs 0..3 are tokenizer control symbols and must never be copied.
        copy_mask = source_mask & source_ids.ge(4)
        attention_scores = attention_scores.masked_fill(
            ~copy_mask.unsqueeze(1), torch.finfo(attention_scores.dtype).min
        )
        attention = attention_scores.softmax(dim=-1)
        copy_probabilities = torch.zeros_like(vocabulary_logits).scatter_add(
            2,
            source_ids.unsqueeze(1).expand(-1, decoded.shape[1], -1),
            attention,
        )
        context = torch.matmul(attention, memory)
        generation_gate = torch.sigmoid(
            self.copy_gate(torch.cat((decoded, context), dim=-1))
        )
        probabilities = (
            generation_gate * vocabulary_logits.softmax(dim=-1)
            + (1.0 - generation_gate) * copy_probabilities
        )
        return probabilities.clamp_min(1e-8).log()

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _decode(
        self,
        decoder_input_ids: torch.Tensor,
        decoder_mask: torch.Tensor,
        memory: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        target = self._embed(
            decoder_input_ids,
            self.target_positions,
            self.config.max_target_length,
        )
        target_length = decoder_input_ids.shape[1]
        causal_mask = torch.triu(
            torch.ones(
                target_length,
                target_length,
                dtype=torch.bool,
                device=decoder_input_ids.device,
            ),
            diagonal=1,
        )
        return self.decoder(
            target,
            memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=~decoder_mask.bool(),
            memory_key_padding_mask=~source_mask.bool(),
        )

    def _embed(
        self,
        token_ids: torch.Tensor,
        positions: nn.Embedding,
        maximum: int,
    ) -> torch.Tensor:
        length = token_ids.shape[1]
        if length > maximum:
            raise ValueError(f"sequence length {length} exceeds configured maximum {maximum}")
        position_ids = torch.arange(length, device=token_ids.device).unsqueeze(0)
        return self.embedding_dropout(self.token_embedding(token_ids) + positions(position_ids))

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[self.config.pad_id].zero_()
        nn.init.normal_(self.source_positions.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.target_positions.weight, mean=0.0, std=0.02)


class _CNNEncoder(nn.Module):
    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        kernels = (3, 5, 7)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(config.d_model),
                _ChannelFirstConv(config.d_model, kernels[index % len(kernels)]),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.d_model, config.d_model),
            )
            for index in range(config.encoder_layers)
        )
        self.output_norm = nn.LayerNorm(config.d_model)

    def forward(self, embedded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        values = embedded
        float_mask = mask.unsqueeze(-1).to(values.dtype)
        for block in self.blocks:
            values = (values + block(values)) * float_mask
        return self.output_norm(values) * float_mask


class _ChannelFirstConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=1,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.conv(values.transpose(1, 2)).transpose(1, 2)


class _BiGRUEncoder(nn.Module):
    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        self.gru = nn.GRU(
            config.d_model,
            config.d_model // 2,
            num_layers=config.encoder_layers,
            dropout=config.dropout if config.encoder_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(config.d_model)

    def forward(self, embedded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        encoded, _ = self.gru(packed)
        values, _ = nn.utils.rnn.pad_packed_sequence(
            encoded,
            batch_first=True,
            total_length=embedded.shape[1],
        )
        return self.output_norm(values) * mask.unsqueeze(-1).to(values.dtype)


class _TransformerEncoder(nn.Module):
    def __init__(self, config: BaselineConfig) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.encoder_layers,
            norm=nn.LayerNorm(config.d_model),
            enable_nested_tensor=False,
        )

    def forward(self, embedded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(embedded, src_key_padding_mask=~mask.bool())
