"""Train a CNN or BiLSTM baseline.

Usage:
    python -m src.train_baseline --config configs/cnn_hasoc.yaml
    python -m src.train_baseline --config configs/lstm_hasoc.yaml
"""
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.build import load_splits
from src.data.datasets import BaselineDataset, Vocab
from src.metrics import compute_metrics
from src.models.cnn import TextCNN
from src.models.lstm import BiLSTMAttention
from src.utils import ensure_dir, get_device, load_config, save_json, set_seed


def build_model(m, vocab_size):
    if m["type"] == "cnn":
        return TextCNN(vocab_size, m["embedding_dim"], m["num_filters"], tuple(m["kernel_sizes"]),
                       m.get("num_labels", 2), m["dropout"])
    if m["type"] == "lstm":
        return BiLSTMAttention(vocab_size, m["embedding_dim"], m["hidden_dim"],
                               m.get("num_labels", 2), m["dropout"])
    raise ValueError(f"Unknown baseline type: {m['type']}")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        logits = model(batch["input_ids"].to(device))
        preds += logits.argmax(dim=1).tolist()
        labels += batch["label"].tolist()
    return compute_metrics(labels, preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = get_device()
    out_dir = ensure_dir(cfg["output_dir"])
    m, t = cfg["model"], cfg["training"]

    train, val, test = load_splits(cfg, seed)
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")

    vocab = Vocab(train, min_freq=m.get("min_freq", 2))
    save_json(vocab.itos, os.path.join(out_dir, "vocab.json"))
    loaders = {
        name: DataLoader(BaselineDataset(data, vocab, m["max_length"]),
                         batch_size=t["batch_size"], shuffle=(name == "train"))
        for name, data in [("train", train), ("val", val), ("test", test)]
    }

    model = build_model(m, len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(t["learning_rate"]))
    criterion = nn.CrossEntropyLoss()

    best_f1, best_val = -1.0, None
    ckpt_path = os.path.join(out_dir, "best_model.pt")
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        total = 0.0
        for batch in tqdm(loaders["train"], desc=f"epoch {epoch}"):
            optimizer.zero_grad()
            logits = model(batch["input_ids"].to(device))
            loss = criterion(logits, batch["label"].to(device))
            loss.backward()
            optimizer.step()
            total += loss.item()
        val_metrics = evaluate(model, loaders["val"], device)
        print(f"epoch {epoch}: train_loss={total / len(loaders['train']):.4f}  val_macro_f1={val_metrics['macro_f1']:.4f}")
        if val_metrics["macro_f1"] > best_f1:
            best_f1, best_val = val_metrics["macro_f1"], val_metrics
            torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_metrics = evaluate(model, loaders["test"], device)
    print("test:", test_metrics)
    save_json({"val": best_val, "test": test_metrics, "config": cfg}, os.path.join(out_dir, "metrics.json"))


if __name__ == "__main__":
    main()
