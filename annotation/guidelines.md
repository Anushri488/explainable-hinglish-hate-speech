# Hinglish rationale annotation guidelines

HASOC has labels but no word-level rationales. We annotate a small sample ourselves so we can measure **plausibility** on code-mixed text.

## Create the file

```bash
python -m src.data.annotation --config configs/muril_hasoc.yaml --n 200 --out annotation/to_annotate.csv
```

## What to mark

For each post (all are labelled HOF), write the **word indices** of the words that make the post hateful or offensive, using the `word_index_guide` column. Separate indices with spaces, e.g. `3 4 9`.

- Mark slurs, insults, abusive words, and words that target a group or person in a hateful way.
- Mark the *smallest* set of words that justifies the label. Do not mark the whole sentence.
- Mark a word even if the spelling is unusual (Hinglish spelling varies a lot).
- Do not mark neutral words (e.g. "yeh", "the", "hai") unless they are part of an abusive phrase.
- If you think the HOF label is wrong, leave the cell empty and add a note in a `notes` column.

## Process

1. Each post is annotated by **two** team members independently (`rationale_annotator1`, `rationale_annotator2`). Do not look at each other's answers.
2. Before the full sample, all four of us annotate the same 20 posts and discuss disagreements, then refine these guidelines.
3. Save the finished file as `annotation/hinglish_rationales.csv`.
4. Report agreement between annotators (e.g. token-level Cohen's kappa) in the final report.

A word counts as a gold rationale if at least half of the annotators marked it.

**Note:** this content can be upsetting. Take breaks, and it is fine to swap posts with a teammate.
