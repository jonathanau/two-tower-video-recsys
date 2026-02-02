from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import hnswlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from two_tower_recsys.data import (
    DataArtifacts,
    TwoTowerExamplesDataset,
    collate_examples,
    prepare_data,
    write_data_artifacts,
)
from two_tower_recsys.metrics import mrr_at_k, recall_at_k
from two_tower_recsys.model import TwoTowerConfig, TwoTowerModel


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


@torch.no_grad()
def evaluate_recall_mrr(
    model: TwoTowerModel,
    examples: list[tuple[int, list[int], int]],
    *,
    num_items: int,
    pad_item_index: int,
    k: int = 10,
    batch_size: int = 512,
    device: str,
) -> tuple[float, float]:
    if not examples:
        return 0.0, 0.0

    model.eval()

    all_item_ids = torch.arange(num_items, device=device, dtype=torch.long)
    item_vecs = model.encode_items(all_item_ids)

    ds = TwoTowerExamplesDataset(examples)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_examples(b, pad_item_index=pad_item_index),
    )

    hits: list[bool] = []
    ranks: list[int] = []

    for user_ids, histories, lengths, targets in dl:
        user_ids = user_ids.to(device)
        histories = histories.to(device)
        lengths = lengths.to(device)
        targets = targets.to(device)

        user_vecs = model.encode_users(user_ids, histories, lengths)
        scores = user_vecs @ item_vecs.T

        for i in range(scores.shape[0]):
            hist = histories[i]
            for it in hist.tolist():
                if it == pad_item_index:
                    continue
                scores[i, it] = -1e9

        topk = torch.topk(scores, k=min(k, num_items), dim=-1).indices

        for i in range(topk.shape[0]):
            t = int(targets[i].item())
            recs = topk[i].tolist()
            if t in recs:
                hits.append(True)
                ranks.append(recs.index(t) + 1)
            else:
                hits.append(False)
                ranks.append(0)

    hit_mask = torch.tensor(hits)
    rank_tensor = torch.tensor(ranks)
    return recall_at_k(hit_mask), mrr_at_k(rank_tensor)


def train_two_tower(
    *,
    data_dir: Path,
    artifacts_dir: Path,
    epochs: int = 5,
    batch_size: int = 256,
    embedding_dim: int = 64,
    max_history_len: int = 50,
    min_rating: float = 4.0,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    device: str = "cpu",
    k_eval: int = 10,
) -> None:
    _set_seed(seed)

    data: DataArtifacts = prepare_data(
        data_dir,
        min_rating=min_rating,
        max_history_len=max_history_len,
    )

    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    write_data_artifacts(artifacts_dir, data)

    config = TwoTowerConfig(
        num_users=len(data.mappings.index_to_user_id),
        num_items=len(data.mappings.index_to_item_id),
        num_genres=len(data.genre_names),
        embedding_dim=embedding_dim,
        max_history_len=max_history_len,
    )

    model = TwoTowerModel(
        config,
        pad_item_index=data.mappings.pad_item_index,
        item_genre_ids_by_index=data.item_genre_ids_by_index,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_ds = TwoTowerExamplesDataset(data.train_examples)
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=lambda b: collate_examples(b, pad_item_index=data.mappings.pad_item_index),
    )

    best_val = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_steps = 0

        pbar = tqdm(train_dl, desc=f"epoch {epoch}/{epochs}")
        for user_ids, histories, lengths, targets in pbar:
            user_ids = user_ids.to(device)
            histories = histories.to(device)
            lengths = lengths.to(device)
            targets = targets.to(device)

            user_vecs = model.encode_users(user_ids, histories, lengths)
            item_vecs = model.encode_items(targets)

            loss = model.in_batch_softmax_loss(user_vecs, item_vecs)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += float(loss.item())
            n_steps += 1
            pbar.set_postfix(loss=f"{total_loss / max(1, n_steps):.4f}")

        val_recall, val_mrr = evaluate_recall_mrr(
            model,
            data.val_examples,
            num_items=config.num_items,
            pad_item_index=data.mappings.pad_item_index,
            k=k_eval,
            device=device,
        )

        if val_recall > best_val:
            best_val = val_recall

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "pad_item_index": data.mappings.pad_item_index,
                },
                artifacts_dir / "model.pt",
            )

        with (artifacts_dir / "train_log.jsonl").open("a") as f:
            f.write(json.dumps({"epoch": epoch, "val_recall": val_recall, "val_mrr": val_mrr}) + "\n")

    ckpt = torch.load(artifacts_dir / "model.pt", map_location=device)
    model2 = TwoTowerModel(
        TwoTowerConfig(**ckpt["config"]),
        pad_item_index=int(ckpt["pad_item_index"]),
        item_genre_ids_by_index=data.item_genre_ids_by_index,
    ).to(device)
    model2.load_state_dict(ckpt["model_state_dict"], strict=True)
    model2.eval()

    with torch.no_grad():
        all_item_ids = torch.arange(config.num_items, device=device, dtype=torch.long)
        item_vecs = model2.encode_items(all_item_ids).cpu().numpy()

    item_vecs = np.asarray(item_vecs, dtype=np.float32)
    np.save(artifacts_dir / "item_embeddings.npy", item_vecs)

    ann = hnswlib.Index(space="ip", dim=int(item_vecs.shape[1]))
    ann.init_index(max_elements=int(item_vecs.shape[0]), ef_construction=200, M=16)
    ann.add_items(item_vecs, ids=np.arange(int(item_vecs.shape[0])))
    ann.set_ef(50)
    ann.save_index(str(artifacts_dir / "hnsw_index.bin"))

    test_recall, test_mrr = evaluate_recall_mrr(
        model2,
        data.test_examples,
        num_items=config.num_items,
        pad_item_index=data.mappings.pad_item_index,
        k=k_eval,
        device=device,
    )

    with (artifacts_dir / "test_metrics.json").open("w") as f:
        json.dump({"recall": test_recall, "mrr": test_mrr, "k": k_eval}, f)
