# Explainable Hate Speech Detection in Code-Mixed Hindi-English

Detect hateful / offensive content in code-mixed Hindi-English (Hinglish) social media posts, **and highlight the words responsible for each decision**.

## Research question

> Does rationale supervision learned from English data (HateXplain) improve the **faithfulness** of explanations on code-mixed Hinglish text?

- **Faithful** explanation: the highlighted words really are what drives the model's prediction.
- **Plausible** explanation: the highlighted words agree with what a human would mark.

## Datasets

| Dataset | Language | Labels | Word-level rationales | Source |
|---|---|---|---|---|
| HASOC (Hindi / Hinglish tracks) | Hindi, code-mixed Hinglish | NOT / HOF | No | https://hasocfire.github.io |
| HateXplain | English | normal / offensive / hatespeech | Yes (human-annotated) | https://github.com/hate-alert/HateXplain |

Both are mapped to a common binary label: **NOT = 0**, **HOF (hateful or offensive) = 1**.
Datasets are **not** committed to this repo. See [`data/README.md`](data/README.md) for download steps.

## Approach

1. **Baselines:** CNN (Kim, 2014) and BiLSTM with attention.
2. **Multilingual transformers:** fine-tune MuRIL and XLM-R on HASOC.
3. **Rationale-supervised training:** train on HateXplain with an extra loss that pushes the model's attention towards human-marked words, then fine-tune on HASOC.
   `loss = cross_entropy + λ · attention_loss`
4. **Explanations:** last-layer [CLS] attention and LIME.
5. **Evaluation:**
   - Classification: accuracy, macro-F1.
   - Faithfulness: comprehensiveness (↑) and sufficiency (↓), as in ERASER / HateXplain.
   - Plausibility: token F1 against a small set of Hinglish posts we annotate ourselves.

### Main comparison

| Model | Stage 1 (HateXplain) | Stage 2 (HASOC) |
|---|---|---|
| MuRIL | — | fine-tune |
| MuRIL + English (control) | fine-tune, λ = 0 | fine-tune |
| MuRIL + English rationales | fine-tune, λ > 0 | fine-tune |

The control row matters: it separates the effect of *rationale supervision* from the effect of simply seeing *more (English) data*.

## Repository structure

```
├── configs/               # one YAML file per experiment (all hyperparameters live here)
├── data/                  # dataset instructions; raw files are git-ignored
├── annotation/            # guidelines + our Hinglish rationale annotations
├── src/
│   ├── data/              # loading, cleaning, PyTorch datasets, annotation helpers
│   ├── models/            # CNN, BiLSTM, transformer classifier
│   ├── losses/            # rationale (attention) supervision loss
│   ├── explain/           # attention, LIME, faithfulness metrics, HTML visualisation
│   ├── train_baseline.py
│   ├── train_transformer.py
│   ├── metrics.py
│   └── utils.py
├── notebooks/             # Colab notebooks for running experiments
├── results/               # metrics and explanation outputs (checkpoints are git-ignored)
├── tests/                 # smoke test on tiny fake data
└── docs/                  # project plan, roles, meeting notes, experiment log
```

## Setup

```bash
git clone <this-repo-url>
cd explainable-hinglish-hate-speech
pip install -r requirements.txt
python -m tests.smoke_test   # quick check that everything runs
```

A GPU is recommended for transformers (Google Colab works; see [`notebooks/README.md`](notebooks/README.md)).

## How to run

All commands are run from the repository root.

```bash
# Baselines
python -m src.train_baseline --config configs/cnn_hasoc.yaml
python -m src.train_baseline --config configs/lstm_hasoc.yaml

# Transformers on HASOC
python -m src.train_transformer --config configs/muril_hasoc.yaml
python -m src.train_transformer --config configs/xlmr_hasoc.yaml

# Rationale experiment (run stage 1 before stage 2)
python -m src.train_transformer --config configs/muril_stage1_hatexplain_rationale.yaml
python -m src.train_transformer --config configs/muril_stage2_hasoc_from_rationale.yaml

# Control (same stages, no rationale supervision)
python -m src.train_transformer --config configs/muril_stage1_hatexplain_plain.yaml
python -m src.train_transformer --config configs/muril_stage2_hasoc_from_plain.yaml

# Explanations + faithfulness
python -m src.explain.run_explanations \
    --config configs/muril_stage2_hasoc_from_rationale.yaml \
    --checkpoint results/muril_stage2_hasoc_from_rationale/best_model.pt
```

Each run writes `metrics.json` (and, for explanations, a summary JSON and an HTML page of highlighted examples) into its `output_dir`.

**To change an experiment, edit its config file, not the code** (e.g. `learning_rate`, `max_length`, `pretrained_name`, `rationale.lambda`).

## Results

_To be filled in as experiments finish. See [`docs/experiment_log.md`](docs/experiment_log.md)._

| Model | Macro-F1 | Comprehensiveness ↑ | Sufficiency ↓ | Token F1 (plausibility) ↑ |
|---|---|---|---|---|
| CNN | | – | – | – |
| BiLSTM + attention | | | | |
| MuRIL | | | | |
| XLM-R | | | | |
| MuRIL + English (control) | | | | |
| MuRIL + English rationales | | | | |

## Team

- Mausam Kumari
- Anushri Mishra
- Manoj Dewda
- Kuldeep Saraswat

Roles will be decided by the team; see [`docs/team_roles.md`](docs/team_roles.md).

## References

1. Mathew et al. *HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection.* AAAI 2021. https://arxiv.org/abs/2012.10289
2. HASOC: Hate Speech and Offensive Content Identification (FIRE shared task). https://hasocfire.github.io
3. Khanuja et al. *MuRIL: Multilingual Representations for Indian Languages.* 2021. https://arxiv.org/abs/2103.10730
4. Conneau et al. *Unsupervised Cross-lingual Representation Learning at Scale (XLM-R).* ACL 2020. https://arxiv.org/abs/1911.02116
5. DeYoung et al. *ERASER: A Benchmark to Evaluate Rationalized NLP Models.* ACL 2020. https://arxiv.org/abs/1911.03429
6. Ribeiro et al. *"Why Should I Trust You?": Explaining the Predictions of Any Classifier (LIME).* KDD 2016. https://arxiv.org/abs/1602.04938
7. Jain & Wallace. *Attention is not Explanation.* NAACL 2019. https://arxiv.org/abs/1902.10186
8. Kim. *Convolutional Neural Networks for Sentence Classification.* EMNLP 2014. https://arxiv.org/abs/1408.5882
