"""
Index and validate available dataset cases.

This module scans the raw dataset directories, matches files by case ID,
and builds a structured index mapping each case to its available modalities
(e.g., ADC, Z-map) and ground-truth labels.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from lesion_segmentation.paths import load_dataset_paths

CASE_ID_RE = re.compile(r"(?P<pid>MGHNICU_\d+)-VISIT_(?P<visit>\d+)")


@dataclass(frozen=True)
class CaseFiles:
    case_id: str
    adc_mha: Path
    label_mha: Optional[Path]
    zmap_mha: Optional[Path]


def parse_case_id_from_adc_filename(name: str) -> Optional[str]:
    """
    Parse case ID from an ADC filename.

    Example:
        MGHNICU_010-VISIT_01-ADC_ss.mha -> MGHNICU_010_VISIT_01
    """
    m = CASE_ID_RE.search(name)
    if not m:
        return None

    pid = m.group(1)
    visit = m.group(2).zfill(2)
    return f"{pid}_VISIT_{visit}"


def find_raw_split_dirs(raw_root: Path) -> Tuple[Path, Path, Path]:
    train = raw_root / "BONBID2023_Train"
    val = raw_root / "BONBID2023_Val"
    test = raw_root / "BONBID2023_Test"
    # Some downloads may have Test as zip only; allow missing test dir for now.
    return train, val, test


def index_split(split_dir: Path) -> Dict[str, CaseFiles]:
    """
    Build a mapping from case_id -> CaseFiles for a given split folder.
    Expected subfolders: 1ADC_ss, 2Z_ADC, 3LABEL (labels may be absent in test).
    """
    adc_dir = split_dir / "1ADC_ss"
    z_dir = split_dir / "2Z_ADC"
    label_dir = split_dir / "3LABEL"

    if not adc_dir.exists():
        raise FileNotFoundError(f"Missing ADC folder: {adc_dir}")

    # ADC
    adc_files = sorted(adc_dir.glob("*.mha"))
    out: Dict[str, CaseFiles] = {}

    # Optional modality sets
    zmap_lookup: Dict[str, Path] = {}
    if z_dir.exists():
        for p in z_dir.glob("*.mha"):
            # Example: Zmap_MGHNICU_010-VISIT_01-ADC_smooth2mm_clipped10.mha
            # Parse embedded "MGHNICU_010-VISIT_01"
            m = CASE_ID_RE.search(p.name)
            if m:
                pid = m.group("pid")
                visit = m.group("visit").zfill(2)
                cid = f"{pid}_VISIT_{visit}"
                zmap_lookup[cid] = p

    label_lookup: Dict[str, Path] = {}
    if label_dir.exists():
        for p in label_dir.glob("*.mha"):
            # Example: MGHNICU_010-VISIT_01_lesion.mha
            m = CASE_ID_RE.match(p.name)
            if m:
                pid = m.group("pid")
                visit = m.group("visit").zfill(2)
                cid = f"{pid}_VISIT_{visit}"
                label_lookup[cid] = p

    for adc in adc_files:
        cid = parse_case_id_from_adc_filename(adc.name)
        if not cid:
            continue
        out[cid] = CaseFiles(
            case_id=cid,
            adc_mha=adc,
            label_mha=label_lookup.get(cid),
            zmap_mha=zmap_lookup.get(cid),
        )

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to dataset config YAML.")
    args = ap.parse_args()

    dp = load_dataset_paths(args.config)
    train_dir, val_dir, test_dir = find_raw_split_dirs(dp.raw_root)

    print(f"[dataset_index] repo_root-resolved raw_root: {dp.raw_root}")

    for name, split in [("Train", train_dir), ("Val", val_dir), ("Test", test_dir)]:
        if not split.exists():
            print(f"[dataset_index] {name}: {split} (missing, skipping)")
            continue
        idx = index_split(split)
        n = len(idx)
        n_with_label = sum(1 for c in idx.values() if c.label_mha is not None)
        n_with_z = sum(1 for c in idx.values() if c.zmap_mha is not None)
        print(f"[dataset_index] {name}: {n} cases | labels: {n_with_label} | z-maps: {n_with_z}")

        # Print a few examples
        for i, (cid, cf) in enumerate(sorted(idx.items())[:3]):
            print(f"  - {cid}: adc={cf.adc_mha.name} label={'yes' if cf.label_mha else 'no'} z={'yes' if cf.zmap_mha else 'no'}")
            if i >= 2:
                break


if __name__ == "__main__":
    main()
