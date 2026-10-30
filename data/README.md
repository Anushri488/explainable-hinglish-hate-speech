# Data

Datasets are **not** stored in this repository (licensing and file size). Every team member downloads them locally into the folders below.

```
data/raw/hasoc/        <- HASOC files
data/raw/hatexplain/   <- dataset.json, post_id_divisions.json
data/processed/        <- anything we generate
```

## HASOC (Hindi / Hinglish)

1. Go to https://hasocfire.github.io and open the page for the year you need.
   - Hindi tracks: HASOC 2019 and 2020.
   - Code-mixed Hinglish: see the code-mixed subtasks of later editions (e.g. 2021).
2. Some editions require a registration / data request form. **Do this early**; approval can take time.
3. Put the files in `data/raw/hasoc/`.
4. Open the files and check the column names. Then update `train_path`, `test_path`, `text_column`, `label_column` and `label_map` in the HASOC configs.
   - Note: some editions store conversations (post → comments → replies) in nested JSON. Those need a small conversion script to a flat table first.

Record exactly which HASOC edition and subtask we use here:

| Edition | Subtask | Files | Train size | Test size |
|---|---|---|---|---|
| | | | | |

## HateXplain (English)

```bash
cd data/raw/hatexplain
wget https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json
wget https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/post_id_divisions.json
```

Label mapping used in this project: `normal -> 0`, `offensive -> 1`, `hatespeech -> 1`.
