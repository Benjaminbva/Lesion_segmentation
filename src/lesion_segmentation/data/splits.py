"""
Define and manage dataset train/validation/test splits.

This module creates, loads, and saves deterministic dataset splits and
subsets to ensure reproducible experiments and prevent data leakage.
"""

from __future__ import annotations

from pathlib import Path
from typing import List


def load_subset_list(path: str | Path) -> List[str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    case_ids: List[str] = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        case_ids.append(s)
    if not case_ids:
        raise ValueError(f"Manifest is empty: {path}")
    return case_ids
