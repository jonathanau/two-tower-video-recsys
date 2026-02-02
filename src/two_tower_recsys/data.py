from __future__ import annotations

import json
import os
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Mappings:
    user_id_to_index: dict[int, int]
    item_id_to_index: dict[int, int]
    index_to_user_id: list[int]
    index_to_item_id: list[int]
    pad_item_index: int


@dataclass(frozen=True)
class DataArtifacts:
    mappings: Mappings
    item_titles_by_index: list[str]
    item_genre_ids_by_index: list[list[int]]
    genre_names: list[str]
    user_histories_by_index: dict[int, list[int]]
    train_examples: list[tuple[int, list[int], int]]
    val_examples: list[tuple[int, list[int], int]]
    test_examples: list[tuple[int, list[int], int]]


def download_movielens_100k(data_dir: Path) -> Path:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_path = data_dir / "ml-100k.zip"
    extract_dir = data_dir / "ml-100k"

    if extract_dir.exists() and (extract_dir / "u.data").exists():
        return extract_dir

    if not zip_path.exists():
        url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
        urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)

    if not (extract_dir / "u.data").exists():
        raise FileNotFoundError(f"Expected MovieLens file not found: {extract_dir / 'u.data'}")

    return extract_dir


def _try_load_movielens_100k(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    base = Path(data_dir)
    if (base / "u.data").exists() and (base / "u.item").exists():
        ml_dir = base
    elif (base / "ml-100k" / "u.data").exists() and (base / "ml-100k" / "u.item").exists():
        ml_dir = base / "ml-100k"
    else:
        raise FileNotFoundError("MovieLens 100k files not found")

    interactions = pd.read_csv(
        ml_dir / "u.data",
        sep="\t",
        header=None,
        names=["user_id", "item_id", "rating", "timestamp"],
    )

    genre_names = [
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children's",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]

    items = pd.read_csv(
        ml_dir / "u.item",
        sep="|",
        header=None,
        encoding="ISO-8859-1",
    )

    item_cols = [
        "item_id",
        "title",
        "release_date",
        "video_release_date",
        "imdb_url",
    ] + [f"genre_{g}" for g in genre_names]

    items.columns = item_cols

    return interactions, items[["item_id", "title"] + [f"genre_{g}" for g in genre_names]], genre_names


def _try_load_movielens_csv(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    base = Path(data_dir)

    ratings_csv = None
    if (base / "ratings.csv").exists():
        ratings_csv = base / "ratings.csv"
    elif (base / "ml-latest-small" / "ratings.csv").exists():
        ratings_csv = base / "ml-latest-small" / "ratings.csv"

    movies_csv = None
    if (base / "movies.csv").exists():
        movies_csv = base / "movies.csv"
    elif (base / "ml-latest-small" / "movies.csv").exists():
        movies_csv = base / "ml-latest-small" / "movies.csv"

    if ratings_csv is None or movies_csv is None:
        raise FileNotFoundError("MovieLens CSV files not found")

    interactions = pd.read_csv(ratings_csv)
    if "timestamp" not in interactions.columns:
        raise ValueError("Expected 'timestamp' column in ratings.csv")

    interactions = interactions.rename(columns={"userId": "user_id", "movieId": "item_id"})

    movies = pd.read_csv(movies_csv).rename(columns={"movieId": "item_id", "title": "title"})

    genre_set: set[str] = set()
    for g in movies.get("genres", "").fillna("").astype(str):
        for part in g.split("|"):
            if part and part != "(no genres listed)":
                genre_set.add(part)

    genre_names = sorted(genre_set)

    item_genres: list[list[int]] = []
    for g in movies.get("genres", "").fillna("").astype(str):
        ids: list[int] = []
        for part in g.split("|"):
            if part and part != "(no genres listed)":
                ids.append(genre_names.index(part))
        item_genres.append(ids)

    items = movies[["item_id", "title"]].copy()
    items["genre_ids"] = item_genres

    return interactions[["user_id", "item_id", "rating", "timestamp"]], items, genre_names


def load_raw_movielens(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    data_dir = Path(data_dir)

    try:
        interactions, items, genre_names = _try_load_movielens_100k(data_dir)
        if "genre_ids" not in items.columns:
            genre_cols = [c for c in items.columns if c.startswith("genre_")]
            genre_ids: list[list[int]] = []
            for _, row in items.iterrows():
                ids = [i for i, c in enumerate(genre_cols) if int(row[c]) == 1]
                genre_ids.append(ids)
            items = items[["item_id", "title"]].copy()
            items["genre_ids"] = genre_ids
        return interactions, items[["item_id", "title", "genre_ids"]], genre_names
    except FileNotFoundError:
        return _try_load_movielens_csv(data_dir)


def build_mappings(interactions: pd.DataFrame) -> Mappings:
    user_ids = sorted(interactions["user_id"].unique().tolist())
    item_ids = sorted(interactions["item_id"].unique().tolist())

    user_id_to_index = {int(u): i for i, u in enumerate(user_ids)}
    item_id_to_index = {int(it): i for i, it in enumerate(item_ids)}

    return Mappings(
        user_id_to_index=user_id_to_index,
        item_id_to_index=item_id_to_index,
        index_to_user_id=[int(u) for u in user_ids],
        index_to_item_id=[int(it) for it in item_ids],
        pad_item_index=len(item_ids),
    )


def _truncate(seq: list[int], max_len: int) -> list[int]:
    if max_len <= 0:
        return []
    if len(seq) <= max_len:
        return seq
    return seq[-max_len:]


def prepare_data(
    data_dir: Path,
    *,
    min_rating: float = 4.0,
    max_history_len: int = 50,
    min_user_interactions: int = 3,
) -> DataArtifacts:
    interactions, items, genre_names = load_raw_movielens(data_dir)

    interactions = interactions[interactions["rating"] >= min_rating].copy()
    interactions["user_id"] = interactions["user_id"].astype(int)
    interactions["item_id"] = interactions["item_id"].astype(int)
    interactions["timestamp"] = interactions["timestamp"].astype(int)

    mappings = build_mappings(interactions)

    interactions["user_idx"] = interactions["user_id"].map(mappings.user_id_to_index)
    interactions["item_idx"] = interactions["item_id"].map(mappings.item_id_to_index)

    interactions = interactions.dropna(subset=["user_idx", "item_idx"]).copy()
    interactions["user_idx"] = interactions["user_idx"].astype(int)
    interactions["item_idx"] = interactions["item_idx"].astype(int)

    items = items[items["item_id"].isin(mappings.item_id_to_index.keys())].copy()
    items["item_idx"] = items["item_id"].map(mappings.item_id_to_index).astype(int)

    items = items.sort_values("item_idx")

    item_titles_by_index = [""] * len(mappings.index_to_item_id)
    item_genre_ids_by_index: list[list[int]] = [[] for _ in range(len(mappings.index_to_item_id))]

    for _, row in items.iterrows():
        idx = int(row["item_idx"])
        item_titles_by_index[idx] = str(row.get("title", ""))
        g = row.get("genre_ids", [])
        if isinstance(g, str):
            try:
                g = json.loads(g)
            except json.JSONDecodeError:
                g = []
        item_genre_ids_by_index[idx] = [int(x) for x in (g or [])]

    by_user = interactions.sort_values(["user_idx", "timestamp"]).groupby("user_idx")

    full_histories: dict[int, list[int]] = {}
    train_histories: dict[int, list[int]] = {}
    val_examples: list[tuple[int, list[int], int]] = []
    test_examples: list[tuple[int, list[int], int]] = []

    for user_idx, df_u in by_user:
        items_u = df_u["item_idx"].astype(int).tolist()
        if len(items_u) < min_user_interactions:
            continue

        full_histories[int(user_idx)] = items_u

        train_items = items_u[:-2]
        val_item = items_u[-2]
        test_item = items_u[-1]

        train_histories[int(user_idx)] = train_items

        val_hist = _truncate(train_items, max_history_len)
        test_hist = _truncate(items_u[:-1], max_history_len)

        if len(val_hist) > 0:
            val_examples.append((int(user_idx), val_hist, int(val_item)))
        if len(test_hist) > 0:
            test_examples.append((int(user_idx), test_hist, int(test_item)))

    train_examples: list[tuple[int, list[int], int]] = []
    for user_idx, items_u in train_histories.items():
        for i in range(1, len(items_u)):
            target = int(items_u[i])
            hist = _truncate(items_u[:i], max_history_len)
            if len(hist) == 0:
                continue
            train_examples.append((int(user_idx), hist, target))

    return DataArtifacts(
        mappings=mappings,
        item_titles_by_index=item_titles_by_index,
        item_genre_ids_by_index=item_genre_ids_by_index,
        genre_names=genre_names,
        user_histories_by_index=full_histories,
        train_examples=train_examples,
        val_examples=val_examples,
        test_examples=test_examples,
    )


class TwoTowerExamplesDataset(Dataset):
    def __init__(self, examples: list[tuple[int, list[int], int]]):
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> tuple[int, list[int], int]:
        return self._examples[idx]


def collate_examples(
    batch: Iterable[tuple[int, list[int], int]],
    *,
    pad_item_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = list(batch)
    user_idx = torch.tensor([r[0] for r in rows], dtype=torch.long)
    target_item_idx = torch.tensor([r[2] for r in rows], dtype=torch.long)

    lengths = torch.tensor([len(r[1]) for r in rows], dtype=torch.long)
    max_len = int(lengths.max().item()) if len(rows) > 0 else 0

    histories = torch.full((len(rows), max_len), int(pad_item_index), dtype=torch.long)
    for i, (_, hist, _) in enumerate(rows):
        if len(hist) == 0:
            continue
        histories[i, : len(hist)] = torch.tensor(hist, dtype=torch.long)

    return user_idx, histories, lengths, target_item_idx


def write_data_artifacts(artifacts_dir: Path, data: DataArtifacts) -> None:
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    with (artifacts_dir / "mappings.json").open("w") as f:
        json.dump(
            {
                "index_to_user_id": data.mappings.index_to_user_id,
                "index_to_item_id": data.mappings.index_to_item_id,
                "pad_item_index": data.mappings.pad_item_index,
            },
            f,
        )

    with (artifacts_dir / "items.json").open("w") as f:
        json.dump(
            {
                "item_titles_by_index": data.item_titles_by_index,
                "item_genre_ids_by_index": data.item_genre_ids_by_index,
                "genre_names": data.genre_names,
            },
            f,
        )

    with (artifacts_dir / "user_histories.json").open("w") as f:
        json.dump({str(k): v for k, v in data.user_histories_by_index.items()}, f)


def read_data_artifacts(artifacts_dir: Path) -> tuple[Mappings, list[str], list[list[int]], list[str], dict[int, list[int]]]:
    artifacts_dir = Path(artifacts_dir)

    with (artifacts_dir / "mappings.json").open("r") as f:
        m = json.load(f)

    index_to_user_id = [int(x) for x in m["index_to_user_id"]]
    index_to_item_id = [int(x) for x in m["index_to_item_id"]]
    user_id_to_index = {int(u): i for i, u in enumerate(index_to_user_id)}
    item_id_to_index = {int(it): i for i, it in enumerate(index_to_item_id)}

    mappings = Mappings(
        user_id_to_index=user_id_to_index,
        item_id_to_index=item_id_to_index,
        index_to_user_id=index_to_user_id,
        index_to_item_id=index_to_item_id,
        pad_item_index=int(m["pad_item_index"]),
    )

    with (artifacts_dir / "items.json").open("r") as f:
        items = json.load(f)

    item_titles_by_index = [str(x) for x in items["item_titles_by_index"]]
    item_genre_ids_by_index = [[int(y) for y in x] for x in items["item_genre_ids_by_index"]]
    genre_names = [str(x) for x in items["genre_names"]]

    with (artifacts_dir / "user_histories.json").open("r") as f:
        uh = json.load(f)

    user_histories_by_index = {int(k): [int(x) for x in v] for k, v in uh.items()}

    return mappings, item_titles_by_index, item_genre_ids_by_index, genre_names, user_histories_by_index
