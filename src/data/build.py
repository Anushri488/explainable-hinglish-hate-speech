"""Build train / val / test example lists from a config."""
from sklearn.model_selection import train_test_split

from src.data.load_hasoc import load_hasoc
from src.data.load_hatexplain import load_hatexplain


def load_splits(cfg, seed=42):
    """Return (train, val, test) lists of examples for the dataset named in the config."""
    d = cfg["data"]
    lowercase = d.get("lowercase", True)

    if d["dataset"] == "hasoc":
        train = load_hasoc(d["train_path"], d["text_column"], d["label_column"], d["label_map"], lowercase)
        test = load_hasoc(d["test_path"], d["text_column"], d["label_column"], d["label_map"], lowercase)
        # HASOC gives only train and test, so we carve a stratified validation set out of train.
        train, val = train_test_split(
            train,
            test_size=d.get("val_size", 0.1),
            random_state=seed,
            stratify=[e["label"] for e in train],
        )
        return train, val, test

    if d["dataset"] == "hatexplain":
        splits = load_hatexplain(d["dataset_path"], d["divisions_path"], lowercase)
        return splits["train"], splits["val"], splits["test"]

    raise ValueError(f"Unknown dataset: {d['dataset']}")
