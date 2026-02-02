# Two-Tower Video Recommender (Retrieval)

MIT License

Minimal, research-oriented implementation of a **two-tower retrieval model** with:

- **Sequential user modeling** via a Transformer encoder over watch history
- **Lightweight content features** via MovieLens genre embeddings on the item side
- **In-batch softmax / contrastive learning** objective with a learnable temperature
- **ANN retrieval** via HNSW (`hnswlib`) for fast top-K recommendation
- **Exact retrieval evaluation** (brute-force dot product over all items) with Recall@K and MRR@K

This repository is intentionally small and readable, meant for rapid iteration and ablations.

## Problem setup

- **Task**: next-item prediction (retrieval) from implicit feedback derived from ratings.
- **Dataset**: MovieLens 100K by default (auto-downloaded).
- **Implicit signal**: interactions filtered to `rating >= min_rating` (default `4.0`).
- **Per-user temporal split** (by timestamp):
  - **Train**: all but last 2 interactions
  - **Validation**: 2nd-to-last item, conditioned on train history
  - **Test**: last item, conditioned on history up to the penultimate item

The training set is expanded into prefix/next-item examples: for each user sequence `(i1, i2, ..., in)` we create pairs `([i1..i_{t-1}], i_t)` for `t >= 2` (truncated to `max_history_len`).

## Model

The model learns normalized embeddings `u` (user/context) and `v` (item), scored by dot product.

### Item tower

- **Item ID embedding**: `E_item(item_id)`
- **Genre embedding sum**: `sum_{g in genres(item)} E_genre(g)`
- **MLP projection**: 2-layer GELU MLP
- **Output**: L2-normalized item embedding `v = normalize(MLP(E_item + sum E_genre))`

### User tower

Inputs:

- **User ID embedding**: `E_user(user_id)`
- **History tokens**: item embeddings (same base embedding as item tower) + learned positional embeddings

Architecture:

- **TransformerEncoder** over the history sequence with padding mask
- **Masked mean pooling** of Transformer outputs
- **Concatenate** pooled history + user-id embedding
- **MLP projection**: 2-layer GELU MLP
- **Output**: L2-normalized user/context embedding `u`

Default hyperparameters (see `TwoTowerConfig`):

- **Embedding dim**: 64
- **Transformer layers/heads**: 2 layers, 4 heads
- **Dropout**: 0.1
- **Temperature**: initialized to 0.07 and learned

## Training objective (in-batch softmax)

For a batch of size `B`, we encode user contexts `{u_i}` and their positive target items `{v_i}`.

Let `S_{ij} = (u_i · v_j) / τ` where `τ` is a learned temperature.

We optimize the symmetric in-batch cross-entropy:

- `L = 0.5 * (CE(S, diag) + CE(S^T, diag))`

This is a common two-tower recipe: positives are matched pairs within the batch; all other items in the batch serve as negatives.

## Evaluation

For each validation/test example:

- **Candidate set**: all items in the dataset (`exact` / brute-force scoring)
- **Seen-item masking**: items present in the conditioning history are removed from the candidate set
- **Metrics**:
  - **Recall@K**: hit-rate of the held-out target in the top-K list
  - **MRR@K**: reciprocal rank of the target if present in top-K (0 otherwise)

Evaluation is implemented in `two_tower_recsys.train.evaluate_recall_mrr`.

Recommendations produced by `two_tower_recsys.retrieval.TwoTowerRecommender` use an HNSW ANN index
(when available) and fall back to brute-force scoring if the index artifact is missing.

## Artifacts

Training writes an `artifacts_dir` (default: `<data_dir>/artifacts`) containing:

- **`model.pt`**: PyTorch checkpoint (`state_dict`, config, pad index)
- **`item_embeddings.npy`**: precomputed item embeddings for fast retrieval
- **`hnsw_index.bin`**: HNSW (`hnswlib`) ANN index over item embeddings
- **`mappings.json`**: raw-id ↔ index mapping metadata + pad index
- **`items.json`**: titles, genre IDs, genre vocabulary
- **`user_histories.json`**: full user histories (by internal user index)
- **`train_log.jsonl`**: per-epoch validation metrics
- **`test_metrics.json`**: final test metrics for the best-validation model

## Installation

### Python

`pyproject.toml` declares `requires-python = ">=3.10"`. In practice, **Python 3.10–3.12** is recommended for the smoothest PyTorch install.

### Option A: venv + pip

```bash
python -m venv .venv
source .venv/bin/activate

# If PyTorch install fails on your platform/Python version,
# install it first using the official instructions: https://pytorch.org/
pip install -r requirements.txt

pip install -e .
```

### Option B: conda (CPU example)

```bash
conda create -n two-tower python=3.11 -y
conda activate two-tower

conda install pytorch cpuonly -c pytorch -y
pip install -r requirements.txt --no-deps
pip install -e . --no-deps
```

## Quickstart

### 1) Train

```bash
two-tower train --data-dir ./data --download-movielens --epochs 5
```

Useful flags:

- **`--device`**: `cpu` (default) or `cuda`
- **`--min-rating`**: rating threshold for implicit positives
- **`--max-history-len`**: truncation length for user histories
- **`--embedding-dim`**: embedding dimension
- **`--k-eval`**: K used for Recall@K / MRR@K evaluation

### 2) Retrieve recommendations

```bash
two-tower recommend --data-dir ./data --user-id 1 --k 10 --exclude-seen
```

## Project structure

```text
src/two_tower_recsys/
  cli.py        - command-line entry point (`two-tower`)
  data.py       - dataset download + preprocessing + artifact IO
  model.py      - two-tower architecture + in-batch softmax loss
  train.py      - training loop + exact retrieval evaluation
  retrieval.py  - artifact loading + top-K recommendation
  metrics.py    - Recall@K and MRR@K
```

## Notes for researchers / extensions

- **ANN retrieval**: implemented with HNSW (`hnswlib`); tune `M` / `ef_construction` / `ef` or swap to FAISS / ScaNN.
- **Negatives**: add sampled or hard negatives beyond in-batch negatives.
- **User encoder**: swap mean pooling for CLS token pooling, attention pooling, or GRU.
- **Item features**: add text (titles) via a language model encoder; add multimodal features.
- **Evaluation protocol**: add multiple held-out items per user, popularity/coverage metrics, calibration, etc.

## References

- https://towardsdatascience.com/scaling-recommender-transformers-to-a-billion-parameters/
- https://netflixtechblog.medium.com/towards-generalizable-and-efficient-large-scale-generative-recommenders-a7db648aa257
- https://netflixtechblog.com/foundation-model-for-personalized-recommendation-1a0bd8e02d39

