"""LIME explanations (Ribeiro et al., 2016).

LIME removes random subsets of words, watches how the prediction changes, and fits a
simple linear model on those changes. Its weights are the importance of each word.
"""
import numpy as np
from lime.lime_text import LimeTextExplainer


def lime_word_scores(words, predict_proba, label, num_samples=500, seed=42):
    """Return one weight per word position. Positive = supports `label`."""
    # bow=False keeps each word position separate, so scores line up with `words`.
    explainer = LimeTextExplainer(class_names=["NOT", "HOF"], split_expression=r"\s+",
                                  bow=False, random_state=seed)
    exp = explainer.explain_instance(" ".join(words), predict_proba, num_features=len(words),
                                     labels=(label,), num_samples=num_samples)
    scores = np.zeros(len(words))
    for idx, weight in exp.as_map()[label]:
        scores[idx] = weight
    return scores
