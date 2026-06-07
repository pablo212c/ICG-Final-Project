from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json_config(path: str | Path, allowed_keys: set[str] | None = None) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a JSON object: {config_path}")

    if allowed_keys is not None:
        unknown = sorted(set(config) - allowed_keys)
        if unknown:
            raise ValueError(f"Unknown config key(s) in {config_path}: {', '.join(unknown)}")

    return config


def merged_defaults(
    base_defaults: dict[str, Any],
    config_path: str | Path | None,
    allowed_keys: set[str],
) -> dict[str, Any]:
    defaults = dict(base_defaults)
    if config_path:
        defaults.update(load_json_config(config_path, allowed_keys=allowed_keys))
    return defaults

