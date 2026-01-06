"""
Convert MetaImage (.mha) volumes to NIfTI (.nii.gz) format.

This module performs format conversion only. It preserves image geometry
(spacing, origin, direction) and does not resample, normalize, or relabel
data. It is framework-agnostic and independent of nnU-Net.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import SimpleITK as sitk

from lesion_segmentation.paths import load_dataset_paths
from lesion_segmentation.data.dataset_index import find_raw_split_dirs, index_split
from lesion_segmentation.data.splits import load_subset_list


def write_niigz(img: sitk.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(img, str(out_path), useCompression=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    dp = load_dataset_paths(args.config)
    dp.converted_root.mkdir(parents=True, exist_ok=True)

    wanted = set(load_subset_list(dp.manifests_path))

    train_dir, val_dir, _ = find_raw_split_dirs(dp.raw_root)
    train_idx = index_split(train_dir) if train_dir.exists() else {}
    val_idx = index_split(val_dir) if val_dir.exists() else {}
    idx = {**train_idx, **val_idx}

    missing = [cid for cid in wanted if cid not in idx]
    if missing:
        raise ValueError(f"Manifest contains case_ids not found in Train/Val: {missing[:10]}")

    for cid in sorted(wanted):
        cf = idx[cid]
        # ADC
        adc_out = dp.converted_root / f"{cid}_ADC.nii.gz"
        if not adc_out.exists():
            img = sitk.ReadImage(str(cf.adc_mha))
            write_niigz(img, adc_out)

        # Label (must exist for train/val usage)
        if cf.label_mha is not None:
            lbl_out = dp.converted_root / f"{cid}_lesion.nii.gz"
            if not lbl_out.exists():
                lbl = sitk.ReadImage(str(cf.label_mha))
                write_niigz(lbl, lbl_out)

        # Optional Z-map
        if dp.use_zmap and cf.zmap_mha is not None:
            z_out = dp.converted_root / f"{cid}_Z.nii.gz"
            if not z_out.exists():
                z = sitk.ReadImage(str(cf.zmap_mha))
                write_niigz(z, z_out)

        print(f"[convert] {cid}: wrote ADC{' + label' if cf.label_mha else ''}{' + Z' if (dp.use_zmap and cf.zmap_mha) else ''}")

    print(f"[convert] Done. Output: {dp.converted_root}")


if __name__ == "__main__":
    main()
