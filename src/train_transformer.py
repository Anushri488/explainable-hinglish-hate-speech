"""Fine-tune MuRIL / XLM-R, optionally with rationale (attention) supervision.

Usage:
    # plain fine-tuning on HASOC
    python -m src.train_transformer --config configs/muril_hasoc.yaml

    # two-stage rationale experiment
    python -m src.train_transformer --config configs/muril_stage1_hatexplain_rationale.yaml
    python -m src.train_transformer --config configs/muril_stage2_hasoc_from_rationale.yaml

Set rationale.lambda in the config:
    0   -> normal fine-tuning (no rationale supervision)
    > 0 -> loss = cross_entropy + lambda * attention_loss
"""
import argparse
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from src.data.build import load_splits
from src.data.datasets import TransformerDataset
from src.losses.rationale_loss import attention_supervision_loss
from src.metrics import compute_metrics
from src.models.transformer import TransformerClassifier
from src.utils import ensure_dir, get_device, load_config, save_json, set_seed


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        logits, _ = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
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
    lam = float(cfg.get("rationale", {}).get("lambda", 0.0))

    train, val, test = load_splits(cfg, seed)
    print(f"train={len(train)}  val={len(val)}  test={len(test)}  lambda={lam}")

    tokenizer = AutoTokenizer.from_pretrained(m["pretrained_name"])
    loaders = {
        name: DataLoader(TransformerDataset(data, tokenizer, m["max_length"]),
                         batch_size=t["batch_size"], shuffle=(name == "train"))
        for name, data in [("train", train), ("val", val), ("test", test)]
    }

    model = TransformerClassifier(m["pretrained_name"], m.get("num_labels", 2), m.get("dropout", 0.1)).to(device)
    if cfg.get("init_checkpoint"):
        # Continue from a model trained in an earlier stage (e.g. on HateXplain).
        model.load_state_dict(torch.load(cfg["init_checkpoint"], map_location=device))
        print(f"Loaded weights from {cfg['init_checkpoint']}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(t["learning_rate"]),
                                  weight_decay=float(t.get("weight_decay", 0.01)))
    total_steps = len(loaders["train"]) * t["epochs"]
    scheduler = get_linear_schedule_with_warmup(optimizer, int(float(t.get("warmup_ratio", 0.1)) * total_steps),
                                                total_steps)

    best_f1, best_val = -1.0, None
    ckpt_path = os.path.join(out_dir, "best_model.pt")
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        sum_ce, sum_att = 0.0, 0.0
        for batch in tqdm(loaders["train"], desc=f"epoch {epoch}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits, cls_attention = model(input_ids, attention_mask)
            ce = F.cross_entropy(logits, labels)
            loss = ce
            if lam > 0:
                att = attention_supervision_loss(cls_attention, batch["rationale"].to(device), attention_mask)
                loss = ce + lam * att
                sum_att += att.item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            sum_ce += ce.item()

        n = len(loaders["train"])
        val_metrics = evaluate(model, loaders["val"], device)
        print(f"epoch {epoch}: ce={sum_ce / n:.4f}  att={sum_att / n:.4f}  val_macro_f1={val_metrics['macro_f1']:.4f}")
        if val_metrics["macro_f1"] > best_f1:
            best_f1, best_val = val_metrics["macro_f1"], val_metrics
            torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_metrics = evaluate(model, loaders["test"], device)
    print("test:", test_metrics)
    save_json({"val": best_val, "test": test_metrics, "config": cfg}, os.path.join(out_dir, "metrics.json"))


if __name__ == "__main__":
    main()
