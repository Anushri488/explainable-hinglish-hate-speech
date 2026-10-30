"""Bidirectional LSTM baseline with additive attention.

The attention weights give a simple word-importance explanation even for the baseline.
"""
import torch
import torch.nn as nn


class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=128, num_labels=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(2 * hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * hidden_dim, num_labels)

    def forward(self, input_ids, return_attention=False):
        mask = input_ids != 0                                   # True for real words
        h, _ = self.lstm(self.embedding(input_ids))             # (batch, seq_len, 2*hidden)
        scores = self.attn(h).squeeze(-1).masked_fill(~mask, -1e9)
        alpha = torch.softmax(scores, dim=1)                    # attention over words
        context = (alpha.unsqueeze(-1) * h).sum(dim=1)          # weighted sum
        logits = self.fc(self.dropout(context))
        return (logits, alpha) if return_attention else logits
