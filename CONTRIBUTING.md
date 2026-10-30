# How we work on this repo

## Golden rules

1. **Never push directly to `main`.** Every change goes through a Pull Request (PR).
2. **Every PR is reviewed by at least one other team member** before merging.
3. **Everyone must understand every part of the code.** The author explains the PR to the reviewer; if the reviewer cannot explain it back, the PR is not ready.
4. **Hyperparameters go in `configs/`, never hard-coded in `src/`.**
5. **Never commit datasets or model checkpoints** (`.gitignore` already blocks them).

## Workflow

```bash
git checkout main
git pull                                   # get the latest code
git checkout -b feature/short-description  # e.g. feature/hasoc-loader
# ... make changes ...
git add <files>
git commit -m "Add HASOC loader for 2021 Hinglish files"
git push -u origin feature/short-description
```

Then open a Pull Request on GitHub, fill in the template, and request a review.

## Branch names

- `feature/...` new functionality
- `fix/...` bug fixes
- `exp/...` experiment configs or notebooks
- `docs/...` documentation only

## Commit messages

Use the imperative mood and say *what* changed: `Add LIME explainer`, `Fix rationale alignment for truncated posts`.

## Logging experiments

After every run, add a row to [`docs/experiment_log.md`](docs/experiment_log.md) with the config file, the commit hash (`git rev-parse --short HEAD`) and the results.
