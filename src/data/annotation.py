"""Create and read our own Hinglish rationale annotations.

HASOC has no word-level rationales, so we annotate a small sample ourselves.
This lets us measure plausibility (agreement with humans) on code-mixed text.

Create a file to annotate:
    python -m src.data.annotation --config configs/muril_hasoc.yaml --n 200 \
        --out annotation/to_annotate.csv

See annotation/guidelines.md for how to fill it in.
"""
import argparse
import random

import pandas as pd

from src.data.build import load_splits
from src.utils import load_config


def prepare(cfg, n, out_path, seed=42):
    _, _, test = load_splits(cfg, seed)
    # Rationales only exist for hateful/offensive posts, so we sample HOF posts.
    hof = [e for e in test if e["label"] == 1]
    random.Random(seed).shuffle(hof)
    rows = []
    for e in hof[:n]:
        rows.append({
            "id": e["id"],
            "text": " ".join(e["words"]),
            "word_index_guide": " ".join(f"{i}:{w}" for i, w in enumerate(e["words"])),
            "label": e["label"],
            "rationale_annotator1": "",
            "rationale_annotator2": "",
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote {len(rows)} rows to {out_path}")


def _indices_to_mask(cell, num_words):
    mask = [0] * num_words
    if isinstance(cell, str) and cell.strip():
        for idx in cell.replace(",", " ").split():
            if idx.isdigit() and int(idx) < num_words:
                mask[int(idx)] = 1
    return mask


def load_gold(path):
    """Read a filled annotation CSV. Columns starting with 'rationale_' hold word indices.

    A word is a gold rationale if at least half of the annotators marked it.
    """
    df = pd.read_csv(path)
    ann_cols = [c for c in df.columns if c.startswith("rationale_")]
    examples = []
    for _, row in df.iterrows():
        words = str(row["text"]).split()
        masks = [_indices_to_mask(row[c], len(words)) for c in ann_cols]
        gold = [int(sum(m[i] for m in masks) >= len(masks) / 2) for i in range(len(words))] if masks else None
        examples.append({"id": row["id"], "words": words, "label": int(row["label"]), "rationale": gold})
    return examples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--out", default="annotation/to_annotate.csv")
    args = parser.parse_args()
    cfg = load_config(args.config)
    prepare(cfg, args.n, args.out, seed=cfg.get("seed", 42))
