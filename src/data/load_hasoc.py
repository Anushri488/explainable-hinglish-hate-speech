"""Loader for HASOC (Hindi / code-mixed Hinglish) data.

HASOC files differ by year (CSV, TSV, Excel, JSON) and use different column names,
so the column names and the label mapping are read from the config file.
"""
import os

import pandas as pd

from src.data.preprocess import clean_text, tokenize_words


def read_table(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext == ".tsv":
        return pd.read_csv(path, sep="\t")
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if ext == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported file type: {ext}")


def load_hasoc(path, text_column, label_column, label_map, lowercase=True):
    """Return a list of examples: {"id", "words", "label", "rationale"}.

    HASOC has no word-level rationales, so "rationale" is None.
    """
    df = read_table(path)
    missing = [c for c in (text_column, label_column) if c not in df.columns]
    if missing:
        raise KeyError(f"Columns {missing} not found in {path}. Available: {list(df.columns)}")

    examples = []
    for i, row in df.iterrows():
        raw_label = str(row[label_column]).strip()
        if raw_label not in label_map:
            continue  # skip rows whose label is not in the mapping
        words = tokenize_words(clean_text(row[text_column], lowercase=lowercase))
        if not words:
            continue
        examples.append({
            "id": f"hasoc_{i}",
            "words": words,
            "label": int(label_map[raw_label]),
            "rationale": None,
        })
    return examples
