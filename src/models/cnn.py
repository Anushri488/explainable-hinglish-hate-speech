"""CNN baseline for sentence classification (Kim, 2014)."""
import torch
import torch.nn as nn


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, num_filters=100,
                 kernel_sizes=(3, 4, 5), num_labels=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        # One 1-D convolution per kernel size; each looks at k consecutive words (n-grams).
        self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, num_filters, k) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_labels)

    def forward(self, input_ids):
        x = self.embedding(input_ids).transpose(1, 2)          # (batch, emb_dim, seq_len)
        pooled = [torch.relu(conv(x)).max(dim=2).values for conv in self.convs]  # max over time
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))  # (batch, num_labels)
