"""Faithfulness and plausibility metrics (ERASER benchmark, DeYoung et al., 2020).

Faithfulness: do the highlighted words really drive the model's decision?
  - Comprehensiveness = p(full text) - p(text with top words REMOVED). Higher is better.
  - Sufficiency       = p(full text) - p(ONLY the top words kept).    Lower is better.
  Both are averaged over several "top k%" sizes (bins), like in ERASER and HateXplain.
  They need no human labels, so they work on Hinglish too.

Plausibility: do the highlighted words match what humans marked?
  - Token F1 between the model's top words and the human rationale.
"""
import numpy as np

BINS = (0.01, 0.05, 0.10, 0.20, 0.50)


def comprehensiveness_sufficiency(words, scores, label, predict_proba, bins=BINS):
    n = len(words)
    order = np.argsort(-np.asarray(scores))
    texts = [" ".join(words)]
    for q in bins:
        k = max(1, int(np.ceil(q * n)))
        top = set(order[:k].tolist())
        texts.append(" ".join(w for i, w in enumerate(words) if i not in top))  # top words removed
        texts.append(" ".join(w for i, w in enumerate(words) if i in top))      # only top words kept
    probs = predict_proba(texts)[:, label]
    full, removed, kept = probs[0], probs[1::2], probs[2::2]
    return {
        "comprehensiveness": float(np.mean(full - removed)),
        "sufficiency": float(np.mean(full - kept)),
    }


def token_f1(scores, gold):
    """Mark the top-k words as predicted rationale (k = number of gold words) and compute F1."""
    gold = np.asarray(gold)
    k = int(gold.sum())
    if k == 0:
        return None
    pred = np.zeros_like(gold)
    pred[np.argsort(-np.asarray(scores))[:k]] = 1
    tp = int((pred & gold).sum())
    if tp == 0:
        return 0.0
    precision, recall = tp / pred.sum(), tp / k
    return float(2 * precision * recall / (precision + recall))
