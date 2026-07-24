"""Neural architectures initialized and trained entirely from scratch."""
from __future__ import annotations

import torch
from torch import nn


class CharCNN(nn.Module):
    """Multi-task character CNN for intent classification and slot tagging."""

    def __init__(
        self,
        vocab_size: int,
        num_intents: int,
        num_slots: int,
        embedding_dim: int = 48,
        channels: int = 48,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(embedding_dim, channels, kernel_size=k, padding=k // 2)
            for k in (3, 5, 7)
        )
        feature_dim = channels * len(self.convolutions)
        self.dropout = nn.Dropout(0.2)
        self.intent_attention = nn.Linear(feature_dim, 1)
        self.intent_head = nn.Linear(feature_dim * 2, num_intents)
        self.slot_head = nn.Linear(feature_dim, num_slots)

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor):
        embedded = self.embedding(input_ids).transpose(1, 2)
        features = torch.cat(
            [torch.relu(conv(embedded)) for conv in self.convolutions], dim=1
        ).transpose(1, 2)
        masked = features.masked_fill(~mask.bool().unsqueeze(-1), -1e4)
        max_pooled = masked.max(dim=1).values
        attention_scores = self.intent_attention(features).squeeze(-1)
        attention_scores = attention_scores.masked_fill(~mask.bool(), -1e4)
        attention = torch.softmax(attention_scores, dim=1).unsqueeze(-1)
        attentive_pooled = (features * attention).sum(dim=1)
        pooled = torch.cat((max_pooled, attentive_pooled), dim=-1)
        return self.intent_head(self.dropout(pooled)), self.slot_head(self.dropout(features))


class SequenceBiGRU(nn.Module):
    """Multi-task bidirectional GRU for character or word token sequences."""

    def __init__(
        self,
        vocab_size: int,
        num_intents: int,
        num_slots: int,
        embedding_dim: int = 48,
        hidden_dim: int = 64,
        pad_id: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_id)
        self.encoder = nn.GRU(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(0.2)
        self.intent_head = nn.Linear(hidden_dim * 2, num_intents)
        self.slot_head = nn.Linear(hidden_dim * 2, num_slots)

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor):
        features, _ = self.encoder(self.embedding(input_ids))
        float_mask = mask.unsqueeze(-1).float()
        pooled = (features * float_mask).sum(dim=1) / float_mask.sum(dim=1).clamp_min(1.0)
        return self.intent_head(self.dropout(pooled)), self.slot_head(self.dropout(features))


def build_model(architecture: str, **kwargs) -> nn.Module:
    if architecture == "char_cnn":
        return CharCNN(**kwargs)
    if architecture in {"bigru", "word_bigru"}:
        return SequenceBiGRU(**kwargs)
    raise ValueError(f"unknown architecture: {architecture}")
