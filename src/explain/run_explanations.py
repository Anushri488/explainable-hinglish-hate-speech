"""Generate explanations and measure faithfulness (and plausibility if gold rationales exist).

Usage:
    python -m src.explain.run_explanations --config configs/muril_hasoc.yaml \
        --checkpoint results/muril_hasoc/best_model.pt --num_examples 200

    # On our own annotated Hinglish sample (adds plausibility):
    python -m src.explain.run_explanations --config configs/muril_hasoc.yaml \
        --checkpoint results/muril_hasoc/best_model.pt --gold annotation/hinglish_rationales.csv

Methods:
    attention -> last-layer [CLS] attention
    lime      -> LIME weights
    random    -> random scores (a sanity-check baseline; real methods must beat it)
"""
import argparse
import os
import random

import numpy as np
from tqdm import tqdm

from src.data.annotation import load_gold
from src.data.build import load_splits
from src.explain.faithfulness import comprehensiveness_sufficiency, token_f1
from src.explain.lime_explain import lime_word_scores
from src.explain.predictor import load_predictor
from src.explain.visualize import write_report
from src.utils import get_device, load_config, save_json, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_examples", type=int, default=200)
    parser.add_argument("--lime_samples", type=int, default=500)
    parser.add_argument("--gold", default=None, help="CSV with our own Hinglish rationales")
    parser.add_argument("--methods", default="attention,lime,random")
    parser.add_argument("--out_name", default="explanations")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = get_device()
    methods = args.methods.split(",")
    predictor = load_predictor(cfg, args.checkpoint, device)
    rng = np.random.default_rng(seed)

    if args.gold:
        examples = load_gold(args.gold)
    else:
        _, _, test = load_splits(cfg, seed)
        examples = random.Random(seed).sample(test, min(args.num_examples, len(test)))

    records, per_method = [], {m: {"comprehensiveness": [], "sufficiency": [], "token_f1": []} for m in methods}
    for e in tqdm(examples, desc="explaining"):
        words = e["words"]
        pred = int(predictor.predict_proba([" ".join(words)])[0].argmax())
        rec = {"id": e["id"], "words": words, "label": e["label"], "pred": pred}
        for m in methods:
            if m == "attention":
                scores = predictor.word_attention(words)
            elif m == "lime":
                scores = lime_word_scores(words, predictor.predict_proba, pred, args.lime_samples, seed)
            elif m == "random":
                scores = rng.random(len(words))
            else:
                raise ValueError(f"Unknown method {m}")
            faith = comprehensiveness_sufficiency(words, scores, pred, predictor.predict_proba)
            per_method[m]["comprehensiveness"].append(faith["comprehensiveness"])
            per_method[m]["sufficiency"].append(faith["sufficiency"])
            if e.get("rationale") is not None:
                f1 = token_f1(scores, e["rationale"])
                if f1 is not None:
                    per_method[m]["token_f1"].append(f1)
            rec[m + "_scores"] = [float(s) for s in scores]
        records.append(rec)

    summary = {m: {k: (float(np.mean(v)) if v else None) for k, v in vals.items()} for m, vals in per_method.items()}
    summary["num_examples"] = len(records)
    print(summary)

    out_dir = cfg["output_dir"]
    save_json(summary, os.path.join(out_dir, f"{args.out_name}_summary.json"))
    save_json(records, os.path.join(out_dir, f"{args.out_name}_records.json"))
    write_report(records[:30], os.path.join(out_dir, f"{args.out_name}_examples.html"),
                 [m for m in methods if m != "random"])


if __name__ == "__main__":
    main()
