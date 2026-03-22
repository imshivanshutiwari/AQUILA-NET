from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class ConfigLoader:
    """Loads and merges YAML configuration files from *config_dir*."""

    _CONFIG_FILES = [
        "cable_config.yaml",
        "diffusion_config.yaml",
        "ekf_config.yaml",
        "federated_config.yaml",
        "gnn_config.yaml",
    ]

    def __init__(self, config_dir: str = "configs") -> None:
        self.config_dir = Path(config_dir)
        self._merged: dict = {}

    def load(self, filename: str) -> dict:
        """Load a single YAML config file.

        Args:
            filename: Basename of the file inside *config_dir*.

        Returns:
            Parsed YAML as a dict; returns ``{}`` if the file is missing.
        """
        path = self.config_dir / filename
        if not path.exists():
            return {}
        with open(path, "r") as fh:
            data = yaml.safe_load(fh)
        return data if isinstance(data, dict) else {}

    def load_all(self) -> dict:
        """Load all five AQUILA config files and merge them.

        Later files override keys from earlier files on collision.

        Returns:
            Merged configuration dict.
        """
        merged: dict = {}
        for fname in self._CONFIG_FILES:
            data = self.load(fname)
            merged.update(data)
        self._merged = merged
        return merged

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from the merged config with an optional default.

        Args:
            key: Top-level config key.
            default: Value returned when *key* is absent.

        Returns:
            Config value or *default*.
        """
        if not self._merged:
            self.load_all()
        return self._merged.get(key, default)
