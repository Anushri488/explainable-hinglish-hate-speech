"""Rationale (attention) supervision loss, following HateXplain (Mathew et al., AAAI 2021).

Total loss = cross_entropy(logits, label) + lambda * attention_loss

The human rationale mask is turned into a target probability distribution over tokens
(normalised so it sums to 1). The attention loss is the cross-entropy between this target
and the model's [CLS] attention, which pushes the model to attend to the words humans marked.
Examples without any rationale (e.g. normal posts, or HASOC posts) are skipped.
"""
import torch


def attention_supervision_loss(cls_attention, rationale, attention_mask, eps=1e-8):
    target = rationale * attention_mask.float()
    target_sum = target.sum(dim=1, keepdim=True)
    has_rationale = target_sum.squeeze(1) > 0
    if not has_rationale.any():
        return cls_attention.new_zeros(())
    target = target[has_rationale] / target_sum[has_rationale]
    pred = cls_attention[has_rationale]
    loss = -(target * torch.log(pred + eps)).sum(dim=1)
    return loss.mean()
