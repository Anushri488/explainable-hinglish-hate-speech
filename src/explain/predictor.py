"""Wraps a trained transformer so explanation methods can call it on raw word lists."""
import numpy as np
import torch
from transformers import AutoTokenizer

from src.models.transformer import TransformerClassifier


class TransformerPredictor:
    def __init__(self, model, tokenizer, device, max_length=128, batch_size=32):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

    def _encode(self, list_of_words):
        # An empty input (all words removed) is replaced by the unknown token.
        list_of_words = [w if w else [self.tokenizer.unk_token] for w in list_of_words]
        return self.tokenizer(list_of_words, is_split_into_words=True, truncation=True,
                              max_length=self.max_length, padding=True, return_tensors="pt")

    @torch.no_grad()
    def predict_proba(self, texts):
        """texts: list of strings (words separated by spaces) -> array (n, num_labels)."""
        probs = []
        for i in range(0, len(texts), self.batch_size):
            enc = self._encode([t.split() for t in texts[i:i + self.batch_size]])
            logits, _ = self.model(enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device))
            probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(probs, axis=0)

    @torch.no_grad()
    def word_attention(self, words):
        """[CLS] attention summed over each word's sub-word tokens -> one score per word."""
        enc = self._encode([words])
        word_ids = enc.word_ids(0)
        _, att = self.model(enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device))
        att = att[0].cpu().numpy()
        scores = np.zeros(len(words))
        for t, w in enumerate(word_ids):
            if w is not None:
                scores[w] += att[t]
        return scores


def load_predictor(cfg, checkpoint, device):
    m = cfg["model"]
    tokenizer = AutoTokenizer.from_pretrained(m["pretrained_name"])
    model = TransformerClassifier(m["pretrained_name"], m.get("num_labels", 2), m.get("dropout", 0.1))
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return TransformerPredictor(model.to(device), tokenizer, device, m["max_length"])
