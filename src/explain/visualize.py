"""Render word-importance scores as an HTML page with highlighted words."""
import html

import numpy as np


def highlight(words, scores):
    scores = np.asarray(scores, dtype=float)
    max_abs = np.abs(scores).max() or 1.0
    spans = []
    for w, s in zip(words, scores):
        alpha = min(abs(s) / max_abs, 1.0)
        color = f"rgba(220,50,50,{alpha:.2f})" if s >= 0 else f"rgba(50,100,220,{alpha:.2f})"
        spans.append(f'<span style="background:{color};padding:2px 3px;border-radius:3px" '
                     f'title="{s:.3f}">{html.escape(w)}</span>')
    return " ".join(spans)


def write_report(records, path, methods):
    """records: list of dicts with 'words', 'label', 'pred', and '<method>_scores'."""
    names = ["NOT", "HOF"]
    parts = ["<html><head><meta charset='utf-8'><title>Explanations</title></head>",
             "<body style='font-family:sans-serif;max-width:900px;margin:auto;line-height:1.9'>",
             "<h1>Explanation examples</h1>",
             "<p>Red = pushes towards the predicted class, blue = pushes against it.</p>"]
    for r in records:
        parts.append(f"<hr><p><b>Gold:</b> {names[r['label']]} &nbsp; <b>Predicted:</b> {names[r['pred']]}</p>")
        for m in methods:
            parts.append(f"<p><b>{m}:</b> {highlight(r['words'], r[m + '_scores'])}</p>")
    parts.append("</body></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
