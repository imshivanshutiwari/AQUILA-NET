from typing import List, Optional

import numpy as np
import torch
from sklearn.metrics import f1_score


class AttributionClassifier:
    """Maps attribution probability vectors to human-readable cause labels."""

    DEFAULT_CLASSES = ["seismic", "anchor_drag", "aging", "sabotage"]

    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names if class_names is not None else list(self.DEFAULT_CLASSES)

    def predict(self, attribution_probs: torch.Tensor) -> List[str]:
        """Return the most likely class name for each node.

        Args:
            attribution_probs: tensor of shape (N, C) where C = len(class_names).

        Returns:
            List of N class-name strings.
        """
        indices = attribution_probs.argmax(dim=-1).cpu().tolist()
        return [self.class_names[i % len(self.class_names)] for i in indices]

    def compute_macro_f1(self, probs, labels) -> float:
        """Compute macro-averaged F1 across all attribution classes.

        Args:
            probs: tensor or array-like of shape (N, C) – predicted probabilities.
            labels: tensor or array-like of shape (N,) – ground-truth class indices.

        Returns:
            Macro F1 score as a float.
        """
        if isinstance(probs, torch.Tensor):
            preds = probs.argmax(dim=-1).detach().cpu().numpy()
        else:
            preds = np.asarray(probs).argmax(axis=-1)

        if isinstance(labels, torch.Tensor):
            labels_np = labels.detach().cpu().numpy().ravel()
        else:
            labels_np = np.asarray(labels).ravel()

        return float(f1_score(labels_np, preds, average="macro", zero_division=0))

    def get_confidence(self, probs: torch.Tensor) -> torch.Tensor:
        """Return the maximum probability (confidence) for each sample.

        Args:
            probs: tensor of shape (N, C).

        Returns:
            Tensor of shape (N,) with the max probability per row.
        """
        return probs.max(dim=-1).values
