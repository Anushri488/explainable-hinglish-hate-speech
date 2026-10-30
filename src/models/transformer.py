"""Transformer classifier (MuRIL or XLM-R) that also returns [CLS] attention.

The [CLS] attention from the last layer (averaged over heads) is used
(a) as an explanation and (b) as the quantity supervised by human rationales.
"""
import torch.nn as nn
from transformers import AutoModel


class TransformerClassifier(nn.Module):
    def __init__(self, pretrained_name, num_labels=2, dropout=0.1):
        super().__init__()
        # "eager" attention is needed so the model can return attention weights.
        self.encoder = AutoModel.from_pretrained(pretrained_name, attn_implementation="eager")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        cls_vector = out.last_hidden_state[:, 0]                # representation of [CLS] / <s>
        logits = self.classifier(self.dropout(cls_vector))
        last_layer = out.attentions[-1]                         # (batch, heads, seq, seq)
        cls_attention = last_layer[:, :, 0, :].mean(dim=1)      # (batch, seq): how much [CLS] looks at each token
        return logits, cls_attention
