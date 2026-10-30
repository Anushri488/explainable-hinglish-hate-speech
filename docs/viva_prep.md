# Viva preparation

## Concepts every member must be able to explain
- Why code-mixed text is hard (spelling variation, mixed scripts, transliteration, informal grammar).
- What MuRIL is and how it differs from XLM-R (MuRIL is trained on Indian languages including transliterated text).
- How fine-tuning works: [CLS] representation -> linear classifier -> cross-entropy.
- What attention is, and why "attention as explanation" is debated (Jain & Wallace, 2019).
- How LIME works (perturb the input, fit a local linear model).
- Faithfulness vs plausibility.
- Comprehensiveness and sufficiency: what they measure and which direction is better.
- Rationale supervision: how the human mask becomes a target distribution and how λ trades off accuracy vs attention.
- Why we need the control experiment (English data without rationales).
- Why macro-F1 and not accuracy (class imbalance).

## Practice changes (do these live in mock vivas)
- Switch MuRIL to XLM-R (`pretrained_name` in the config).
- Change λ to 0 and explain why that equals plain fine-tuning.
- Change `max_length` and explain what happens to long posts.
- Show the top-10 words instead of top-k in the HTML report.
- Keep HateXplain's 3 classes instead of merging into 2 (what has to change?).
- Add class weights to the loss for imbalanced data.
- Add a new cleaning step to `clean_text` (e.g. remove "rt" at the start of tweets).
- Replace mean-over-heads attention with a single head, or the second-to-last layer.

## Code walkthrough questions
- Follow one example from a raw CSV row to a prediction: which functions does it pass through?
- In `TransformerDataset`, how is a word-level rationale aligned to sub-word tokens?
- In `attention_supervision_loss`, why are examples without rationales skipped?
- In `comprehensiveness_sufficiency`, what texts are sent to the model for each bin?
