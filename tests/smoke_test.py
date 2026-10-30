"""Quick check that the code runs end to end on tiny fake data (no downloads needed).

Usage (from the repo root):
    python -m tests.smoke_test
"""
import os
import subprocess
import sys
import tempfile

import pandas as pd
import torch
import yaml

from src.losses.rationale_loss import attention_supervision_loss


def test_rationale_loss():
    att = torch.tensor([[0.1, 0.7, 0.2], [0.3, 0.3, 0.4]])
    rationale = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])  # second example has no rationale
    mask = torch.ones(2, 3)
    loss = attention_supervision_loss(att, rationale, mask)
    expected = -torch.log(torch.tensor(0.7))
    assert torch.isclose(loss, expected, atol=1e-4), (loss, expected)
    print("rationale loss OK:", round(loss.item(), 4))


def test_cnn_baseline():
    tmp = tempfile.mkdtemp()
    csv = os.path.join(tmp, "fake.csv")
    pd.DataFrame({"text": ["tu bahut bekar hai", "aaj accha din hai"] * 50,
                  "task_1": ["HOF", "NOT"] * 50}).to_csv(csv, index=False)
    with open("configs/cnn_hasoc.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["data"].update(train_path=csv, test_path=csv)
    cfg["training"]["epochs"] = 2
    cfg["model"]["min_freq"] = 1
    cfg["output_dir"] = os.path.join(tmp, "out")
    cfg_path = os.path.join(tmp, "cfg.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)
    subprocess.run([sys.executable, "-m", "src.train_baseline", "--config", cfg_path], check=True)
    assert os.path.exists(os.path.join(cfg["output_dir"], "metrics.json"))
    print("CNN baseline OK")


if __name__ == "__main__":
    test_rationale_loss()
    test_cnn_baseline()
    print("All smoke tests passed.")
