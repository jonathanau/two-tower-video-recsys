from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from two_tower_recsys.data import read_data_artifacts
from two_tower_recsys.model import TwoTowerConfig, TwoTowerModel


class TwoTowerRecommender:
    def __init__(
        self,
        *,
        artifacts_dir: Path,
        device: str = "cpu",
    ) -> None:
        artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir = artifacts_dir
        self.device = device

        (
            self.mappings,
            self.item_titles_by_index,
            self.item_genre_ids_by_index,
            self.genre_names,
            self.user_histories_by_index,
        ) = read_data_artifacts(artifacts_dir)

        ckpt = torch.load(artifacts_dir / "model.pt", map_location=device)
        cfg = TwoTowerConfig(**ckpt["config"])

        self.model = TwoTowerModel(
            cfg,
            pad_item_index=int(ckpt["pad_item_index"]),
            item_genre_ids_by_index=self.item_genre_ids_by_index,
        ).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.model.eval()

        item_vecs = np.load(artifacts_dir / "item_embeddings.npy")
        self.item_vecs = torch.from_numpy(item_vecs).to(device)

    def recommend(
        self,
        *,
        raw_user_id: int,
        k: int = 10,
        max_history_len: int | None = None,
        exclude_seen: bool = True,
    ) -> list[tuple[int, str, float]]:
        if raw_user_id not in self.mappings.user_id_to_index:
            raise KeyError(f"Unknown user_id={raw_user_id}")

        user_idx = self.mappings.user_id_to_index[raw_user_id]

        hist = self.user_histories_by_index.get(user_idx, [])
        if max_history_len is None:
            max_history_len = self.model.config.max_history_len

        hist = hist[-max_history_len:]
        if len(hist) == 0:
            raise ValueError(f"User {raw_user_id} has no history")

        histories = torch.tensor([hist], dtype=torch.long, device=self.device)
        lengths = torch.tensor([len(hist)], dtype=torch.long, device=self.device)
        user_ids = torch.tensor([user_idx], dtype=torch.long, device=self.device)

        with torch.no_grad():
            user_vec = self.model.encode_users(user_ids, histories, lengths)
            scores = (user_vec @ self.item_vecs.T).squeeze(0)

            if exclude_seen:
                for it in hist:
                    scores[int(it)] = -1e9

            topk = torch.topk(scores, k=min(k, scores.shape[0])).indices.tolist()

        out: list[tuple[int, str, float]] = []
        for item_idx in topk:
            raw_item_id = self.mappings.index_to_item_id[item_idx]
            title = self.item_titles_by_index[item_idx] if item_idx < len(self.item_titles_by_index) else ""
            score = float(scores[item_idx].item())
            out.append((int(raw_item_id), str(title), score))

        return out
