from typing import Dict, Optional, Tuple

import numpy as np


class AugmentationPipeline:
    """Orchestrates DDPM-based dataset augmentation for the sabotage class."""

    _CLASS_NAMES = {0: "normal", 1: "sabotage", 2: "fault", 3: "degradation"}

    def __init__(self, ddpm, otdr_sim, config: Optional[dict] = None):
        self.ddpm = ddpm
        self.otdr_sim = otdr_sim
        self.config = config or {"n_sabotage_augment": 5000}
        self._before_dist: Optional[Dict] = None
        self._after_dist: Optional[Dict] = None

    def augment_dataset(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Augment *X* / *y* by generating additional sabotage samples via DDPM.

        The sabotage class (label == 1) is up-sampled so that it is balanced
        with the most frequent class.  Additional samples are generated with
        the DDPM model and appended to the dataset.

        Returns:
            (X_aug, y_aug) — augmented feature matrix and label vector.
        """
        self._before_dist = self.get_class_distribution(y)

        # Determine how many sabotage samples to generate
        n_augment = self.config.get("n_sabotage_augment", 5000)
        sabotage_label = 1

        # Generate synthetic sabotage traces
        seq_len = X.shape[1] if X.ndim == 2 else None
        synthetic = self.ddpm.generate_sabotage_samples(
            n_samples=n_augment, seq_len=seq_len
        )  # (n_augment, seq_len)

        synthetic_labels = np.full(n_augment, sabotage_label, dtype=y.dtype)

        X_aug = np.concatenate([X, synthetic], axis=0)
        y_aug = np.concatenate([y, synthetic_labels], axis=0)

        self._after_dist = self.get_class_distribution(y_aug)
        return X_aug, y_aug

    def get_class_distribution(self, y: np.ndarray) -> Dict[str, int]:
        """Return {class_name: count} for each unique label in *y*."""
        unique, counts = np.unique(y, return_counts=True)
        return {
            self._CLASS_NAMES.get(int(lbl), str(int(lbl))): int(cnt)
            for lbl, cnt in zip(unique, counts)
        }

    def plot_before_after(self) -> Dict[str, Dict[str, int]]:
        """Return class distribution dicts before and after augmentation.

        Requires that *augment_dataset* has been called at least once.
        """
        if self._before_dist is None or self._after_dist is None:
            raise RuntimeError(
                "augment_dataset() must be called before plot_before_after()."
            )
        return {"before": self._before_dist, "after": self._after_dist}
