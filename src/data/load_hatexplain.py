"""Loader for HateXplain (English, with human word-level rationales).

Expected files (from https://github.com/hate-alert/HateXplain, folder Data/):
    dataset.json            post_id -> {post_tokens, annotators, rationales}
    post_id_divisions.json  {"train": [...], "val": [...], "test": [...]}
"""
import json
from collections import Counter

import numpy as np

# HateXplain has 3 classes. We merge "hatespeech" and "offensive" into one class (HOF = 1)
# so that the label space matches HASOC (NOT = 0 vs HOF = 1).
LABEL_MAP = {"normal": 0, "offensive": 1, "hatespeech": 1}


def majority_label(annotators):
    """Majority vote over the 3 annotators. Returns None if all three disagree."""
    votes = Counter(a["label"] for a in annotators)
    label, count = votes.most_common(1)[0]
    return label if count >= 2 else None


def aggregate_rationale(rationales, num_words, threshold=0.5):
    """Average the annotators' 0/1 masks. A word is a rationale if >= threshold of them marked it."""
    if not rationales:
        return [0] * num_words
    masks = np.array([(list(r) + [0] * num_words)[:num_words] for r in rationales], dtype=float)
    return (masks.mean(axis=0) >= threshold).astype(int).tolist()


def load_hatexplain(dataset_path, divisions_path, lowercase=True):
    """Return a dict: split name -> list of examples {"id", "words", "label", "rationale"}."""
    with open(dataset_path, encoding="utf-8") as f:
        data = json.load(f)
    with open(divisions_path, encoding="utf-8") as f:
        divisions = json.load(f)

    splits = {}
    for split, ids in divisions.items():
        examples = []
        for pid in ids:
            post = data[pid]
            label_name = majority_label(post["annotators"])
            if label_name is None:
                continue  # HateXplain also drops posts with no majority
            words = post["post_tokens"]
            if lowercase:
                words = [w.lower() for w in words]
            examples.append({
                "id": pid,
                "words": words,
                "label": LABEL_MAP[label_name],
                "rationale": aggregate_rationale(post.get("rationales", []), len(words)),
            })
        splits[split] = examples
    return splits
