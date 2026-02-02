from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TwoTowerConfig:
    num_users: int
    num_items: int
    num_genres: int
    embedding_dim: int = 64
    max_history_len: int = 50
    transformer_layers: int = 2
    transformer_heads: int = 4
    transformer_ff_multiplier: int = 4
    dropout: float = 0.1
    temperature: float = 0.07


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        config: TwoTowerConfig,
        *,
        pad_item_index: int,
        item_genre_ids_by_index: list[list[int]] | None = None,
    ) -> None:
        super().__init__()

        self.config = config
        self.pad_item_index = int(pad_item_index)

        self.user_id_embedding = nn.Embedding(config.num_users, config.embedding_dim)
        self.item_id_embedding = nn.Embedding(config.num_items + 1, config.embedding_dim, padding_idx=self.pad_item_index)

        self.pad_genre_index = config.num_genres
        self.genre_embedding = nn.Embedding(config.num_genres + 1, config.embedding_dim, padding_idx=self.pad_genre_index)

        if item_genre_ids_by_index is None:
            item_genre_ids_by_index = [[] for _ in range(config.num_items)]

        max_g = max((len(x) for x in item_genre_ids_by_index), default=0)
        max_g = max(max_g, 1)

        item_genres = torch.full((config.num_items + 1, max_g), self.pad_genre_index, dtype=torch.long)
        for item_idx in range(config.num_items):
            g = item_genre_ids_by_index[item_idx] if item_idx < len(item_genre_ids_by_index) else []
            if not g:
                continue
            gg = torch.tensor(g[:max_g], dtype=torch.long)
            item_genres[item_idx, : gg.numel()] = gg
        item_genres[self.pad_item_index, :] = self.pad_genre_index

        self.register_buffer("item_genres", item_genres, persistent=True)

        self.history_pos_embedding = nn.Embedding(config.max_history_len, config.embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.embedding_dim * config.transformer_ff_multiplier,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.history_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.transformer_layers)

        self.user_mlp = nn.Sequential(
            nn.Linear(config.embedding_dim * 2, config.embedding_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )

        self.item_mlp = nn.Sequential(
            nn.Linear(config.embedding_dim, config.embedding_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.embedding_dim, config.embedding_dim),
        )

        self.log_temperature = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(config.temperature)))))

    def _item_base_embedding(self, item_ids: torch.Tensor) -> torch.Tensor:
        id_emb = self.item_id_embedding(item_ids)
        genre_ids = self.item_genres[item_ids]
        g_emb = self.genre_embedding(genre_ids).sum(dim=-2)
        return id_emb + g_emb

    def encode_items(self, item_ids: torch.Tensor) -> torch.Tensor:
        x = self._item_base_embedding(item_ids)
        x = self.item_mlp(x)
        return torch.nn.functional.normalize(x, dim=-1)

    def encode_users(
        self,
        user_ids: torch.Tensor,
        histories: torch.Tensor,
        history_lengths: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len = histories.shape

        pos = torch.arange(seq_len, device=histories.device).clamp_max(self.config.max_history_len - 1)
        pos = pos.unsqueeze(0).expand(bsz, seq_len)

        hist_emb = self._item_base_embedding(histories) + self.history_pos_embedding(pos)

        key_padding_mask = histories.eq(self.pad_item_index)
        enc = self.history_encoder(hist_emb, src_key_padding_mask=key_padding_mask)

        valid = (~key_padding_mask).float().unsqueeze(-1)
        denom = valid.sum(dim=1).clamp_min(1.0)
        pooled = (enc * valid).sum(dim=1) / denom

        uid = self.user_id_embedding(user_ids)
        x = torch.cat([uid, pooled], dim=-1)
        x = self.user_mlp(x)
        return torch.nn.functional.normalize(x, dim=-1)

    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature).clamp_min(1e-4)

    def in_batch_softmax_loss(
        self,
        user_vecs: torch.Tensor,
        item_vecs: torch.Tensor,
    ) -> torch.Tensor:
        logits = (user_vecs @ item_vecs.T) / self.temperature()
        targets = torch.arange(logits.shape[0], device=logits.device)
        loss_ui = torch.nn.functional.cross_entropy(logits, targets)
        loss_iu = torch.nn.functional.cross_entropy(logits.T, targets)
        return 0.5 * (loss_ui + loss_iu)
