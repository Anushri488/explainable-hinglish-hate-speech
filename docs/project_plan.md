# Project plan

Dates to be filled in once the deadline is confirmed.

| Phase | Goal | Done when | Target date | Status |
|---|---|---|---|---|
| 0. Setup | Repo, environment, everyone reads the HateXplain paper | Everyone can run `pip install -r requirements.txt` | | |
| 1. Data | HASOC + HateXplain loaded with a common label space | `load_splits` works for both datasets | | |
| 2. Baselines | CNN and BiLSTM results on HASOC | `metrics.json` for both | | |
| 3. Transformers | MuRIL and XLM-R fine-tuned on HASOC | `metrics.json` for both | | |
| 4. Rationale supervision | Stage 1 + stage 2 runs, with and without rationales | 4 runs logged | | |
| 5. Explanations | Attention + LIME + faithfulness for all models | Summary JSONs + HTML examples | | |
| 6. Annotation | ~200 Hinglish posts with rationales | `annotation/hinglish_rationales.csv` | | |
| 7. Analysis & report | Answer the research question | Report + slides | | |

## Ideas if time permits
- Tune λ (0.1, 1, 10, 100) and plot faithfulness vs λ.
- Try other explanation methods (e.g. Integrated Gradients).
- Error analysis: which kinds of Hinglish posts the model gets wrong.
