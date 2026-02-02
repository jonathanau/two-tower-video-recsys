from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import hnswlib
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
        self.item_vecs = np.asarray(item_vecs, dtype=np.float32)

        self.ann: hnswlib.Index | None = None
        index_path = artifacts_dir / "hnsw_index.bin"
        if index_path.exists():
            ann = hnswlib.Index(space="ip", dim=int(self.item_vecs.shape[1]))
            ann.load_index(str(index_path), max_elements=int(self.item_vecs.shape[0]))
            ann.set_ef(50)
            self.ann = ann

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
            user_vec_t = self.model.encode_users(user_ids, histories, lengths)

        user_vec = user_vec_t.squeeze(0).detach().cpu().numpy().astype(np.float32)

        seen: set[int] = set(hist) if exclude_seen else set()
        num_items = int(self.item_vecs.shape[0])
        k = int(min(k, num_items))

        item_indices: list[int] = []
        if self.ann is not None:
            self.ann.set_ef(max(50, k * 10))
            requested = int(min(num_items, max(k * 5, k + len(seen))))
            while True:
                labels, _ = self.ann.knn_query(user_vec.reshape(1, -1), k=requested)
                item_indices = []
                for idx in labels[0].tolist():
                    ii = int(idx)
                    if ii < 0:
                        continue
                    if ii in seen:
                        continue
                    item_indices.append(ii)
                    if len(item_indices) >= k:
                        break

                if len(item_indices) >= k or requested >= num_items:
                    break
                requested = int(min(num_items, requested * 2))

        if len(item_indices) < k:
            scores_all = self.item_vecs @ user_vec
            if seen:
                scores_all[list(seen)] = -1e9

            if k == 0:
                return []

            topk = np.argpartition(-scores_all, kth=min(k - 1, num_items - 1))[:k]
            topk = topk[np.argsort(-scores_all[topk])]
            item_indices = [int(x) for x in topk.tolist()]

        scores_k = self.item_vecs[item_indices] @ user_vec

        out: list[tuple[int, str, float]] = []
        for item_idx, score in zip(item_indices, scores_k.tolist()):
            raw_item_id = self.mappings.index_to_item_id[item_idx]
            title = self.item_titles_by_index[item_idx] if item_idx < len(self.item_titles_by_index) else ""
            out.append((int(raw_item_id), str(title), float(score)))

        return out
