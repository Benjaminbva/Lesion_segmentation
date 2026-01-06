"""
Project path resolution utilities.

Resolves config-defined (relative) paths against the repository root so code
does not depend on the current working directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


def repo_root() -> Path:
    # paths.py -> lesion_segmentation -> src -> repo_root
    return Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping/dict, got: {type(data)}")
    return data


def resolve_from_repo(p: str | Path) -> Path:
    p = Path(p)
    return (repo_root() / p).resolve() if not p.is_absolute() else p


@dataclass(frozen=True)
class DatasetPaths:
    raw_root: Path
    converted_root: Path
    nnunet_raw_root: Path

    manifests_type: str
    manifests_path: Path

    use_zmap: bool


def load_dataset_paths(config_path: str | Path) -> DatasetPaths:
    cfg = load_yaml(config_path)

    raw_root = resolve_from_repo(cfg["raw_root"])
    converted_root = resolve_from_repo(cfg["converted_root"])
    nnunet_raw_root = resolve_from_repo(cfg["nnunet_raw_root"])

    manifest = cfg.get("manifest", {})
    manifests_type = manifest.get("type", "subset")
    manifests_path = resolve_from_repo(manifest["path"])

    use_zmap = bool(cfg.get("use_zmap", False))

    return DatasetPaths(
        raw_root=raw_root,
        converted_root=converted_root,
        nnunet_raw_root=nnunet_raw_root,
        manifests_type=manifests_type,
        manifests_path=manifests_path,
        use_zmap=use_zmap,
    )
