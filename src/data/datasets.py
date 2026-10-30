"""PyTorch datasets for the baseline (word-level) and transformer (sub-word) models."""
from collections import Counter

import torch
from torch.utils.data import Dataset

PAD, UNK = "<pad>", "<unk>"


class Vocab:
    """Word vocabulary for the CNN / LSTM baselines. Index 0 = padding, 1 = unknown word."""

    def __init__(self, examples=None, min_freq=2, itos=None):
        if itos is not None:
            self.itos = itos
        else:
            counts = Counter(w for e in examples for w in e["words"])
            self.itos = [PAD, UNK] + sorted(w for w, c in counts.items() if c >= min_freq)
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, words, max_length):
        ids = [self.stoi.get(w, 1) for w in words[:max_length]]
        return ids + [0] * (max_length - len(ids))


class BaselineDataset(Dataset):
    def __init__(self, examples, vocab, max_length):
        self.examples = examples
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        e = self.examples[i]
        return {
            "input_ids": torch.tensor(self.vocab.encode(e["words"], self.max_length)),
            "label": torch.tensor(e["label"]),
        }


class TransformerDataset(Dataset):
    """Tokenises pre-split words and aligns word-level rationales to sub-word tokens.

    Example: the word "bewakoof" may become the sub-words ["bewa", "##koof"]. If the word
    is a rationale (1), both sub-words get 1. Special tokens ([CLS], [SEP], padding) get 0.
    """

    def __init__(self, examples, tokenizer, max_length):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        e = self.examples[i]
        enc = self.tokenizer(
            e["words"],
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        word_ids = enc.word_ids()  # token position -> word index (None for special tokens)
        token_rationale = [0.0] * len(word_ids)
        if e["rationale"] is not None:
            for t, w in enumerate(word_ids):
                if w is not None and e["rationale"][w] == 1:
                    token_rationale[t] = 1.0
        return {
            "input_ids": torch.tensor(enc["input_ids"]),
            "attention_mask": torch.tensor(enc["attention_mask"]),
            "label": torch.tensor(e["label"]),
            "rationale": torch.tensor(token_rationale),
        }
