"""Tiny config loader.

We deliberately avoid a heavy config framework (Hydra, OmegaConf, ...) — the
team is new to this kind of pipeline, and a plain YAML -> dict -> dot-access
object is easy for everyone to read and debug. Load once per script with
`load_config`, and override individual values from the CLI with `--set`
(see any script in scripts/ for the pattern).
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """A dict that also supports attribute access, recursively.

    Config().data.cache_dir works the same as Config()["data"]["cache_dir"].
    """

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[name] = value
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config file into a Config object.

    `overrides` keys use dotted paths, e.g. {"train.batch_size": 16}.
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    cfg = Config(copy.deepcopy(raw))
    if overrides:
        for dotted_key, value in overrides.items():
            _set_dotted(cfg, dotted_key, value)
    return cfg


def _set_dotted(cfg: dict, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = cfg
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def parse_cli_overrides(pairs: list[str]) -> dict[str, Any]:
    """Parse ["train.batch_size=16", "model.dropout=0.1"] into a dict.

    Values are parsed with yaml.safe_load so ints/floats/bools come through
    as the right type, not strings.
    """
    overrides: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Bad override '{pair}', expected key=value")
        key, raw_value = pair.split("=", 1)
        overrides[key.strip()] = yaml.safe_load(raw_value)
    return overrides
