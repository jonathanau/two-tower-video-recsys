from __future__ import annotations

import argparse
from pathlib import Path

from two_tower_recsys.data import download_movielens_100k
from two_tower_recsys.retrieval import TwoTowerRecommender
from two_tower_recsys.train import train_two_tower


def _default_artifacts_dir(data_dir: Path) -> Path:
    return Path(data_dir) / "artifacts"


def _cmd_train(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    if args.download_movielens:
        download_movielens_100k(data_dir)

    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else _default_artifacts_dir(data_dir)

    train_two_tower(
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        max_history_len=args.max_history_len,
        min_rating=args.min_rating,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=args.device,
        k_eval=args.k_eval,
    )


def _cmd_recommend(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    artifacts_dir = Path(args.artifacts_dir) if args.artifacts_dir else _default_artifacts_dir(data_dir)

    rec = TwoTowerRecommender(artifacts_dir=artifacts_dir, device=args.device)
    results = rec.recommend(
        raw_user_id=int(args.user_id),
        k=int(args.k),
        max_history_len=args.max_history_len,
        exclude_seen=bool(args.exclude_seen),
    )

    for rank, (raw_item_id, title, score) in enumerate(results, start=1):
        print(f"{rank:>2}. item_id={raw_item_id}  score={score:.4f}  title={title}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="two-tower")
    sub = p.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--data-dir", type=str, required=True)
    p_train.add_argument("--artifacts-dir", type=str, default=None)
    p_train.add_argument("--download-movielens", action="store_true")
    p_train.add_argument("--epochs", type=int, default=5)
    p_train.add_argument("--batch-size", type=int, default=256)
    p_train.add_argument("--embedding-dim", type=int, default=64)
    p_train.add_argument("--max-history-len", type=int, default=50)
    p_train.add_argument("--min-rating", type=float, default=4.0)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--weight-decay", type=float, default=1e-4)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--device", type=str, default="cpu")
    p_train.add_argument("--k-eval", type=int, default=10)
    p_train.set_defaults(func=_cmd_train)

    p_rec = sub.add_parser("recommend")
    p_rec.add_argument("--data-dir", type=str, required=True)
    p_rec.add_argument("--artifacts-dir", type=str, default=None)
    p_rec.add_argument("--user-id", type=int, required=True)
    p_rec.add_argument("--k", type=int, default=10)
    p_rec.add_argument("--max-history-len", type=int, default=None)
    p_rec.add_argument("--exclude-seen", action="store_true")
    p_rec.add_argument("--device", type=str, default="cpu")
    p_rec.set_defaults(func=_cmd_recommend)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
