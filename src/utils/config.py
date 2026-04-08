import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

import yaml


class Config(SimpleNamespace):
    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def _to_namespace(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Config(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def load_config(config_path: str) -> Config:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping/dict")

    cfg = _to_namespace(data)
    cfg.config_path = str(path)
    return cfg


def parse_train_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fundus training entry")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    return parser.parse_args()
