# Two-Tower Video Recommender (Retrieval)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Train

```bash
two-tower train --data-dir ./data --epochs 5
```

## Recommend

```bash
two-tower recommend --data-dir ./data --user-id 1 --k 10
```
