"""
Verify geometric consistency across dataset volumes.

This utility checks that all modalities and corresponding label masks
share identical spatial geometry (shape, spacing, origin, direction),
preventing silent errors during training and evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import SimpleITK as sitk

from lesion_segmentation.paths import load_dataset_paths
from lesion_segmentation.data.splits import load_subset_list


def geom_key(img: sitk.Image) -> tuple:
    return (
        img.GetSize(),
        img.GetSpacing(),
        img.GetOrigin(),
        img.GetDirection(),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    dp = load_dataset_paths(args.config)
    case_ids = load_subset_list(dp.manifests_path)

    for cid in case_ids:
        adc_p = dp.converted_root / f"{cid}_ADC.nii.gz"
        lbl_p = dp.converted_root / f"{cid}_lesion.nii.gz"
        z_p = dp.converted_root / f"{cid}_Z.nii.gz"

        if not adc_p.exists():
            raise FileNotFoundError(f"Missing converted ADC: {adc_p}")
        if not lbl_p.exists():
            raise FileNotFoundError(f"Missing converted label: {lbl_p}")

        adc = sitk.ReadImage(str(adc_p))
        lbl = sitk.ReadImage(str(lbl_p))

        g_adc = geom_key(adc)
        g_lbl = geom_key(lbl)

        if g_adc != g_lbl:
            raise ValueError(f"[geometry] MISMATCH {cid}: ADC geometry != label geometry")

        if dp.use_zmap:
            if not z_p.exists():
                raise FileNotFoundError(f"Missing converted Z-map: {z_p}")
            z = sitk.ReadImage(str(z_p))
            if geom_key(z) != g_adc:
                raise ValueError(f"[geometry] MISMATCH {cid}: Z geometry != ADC geometry")

        print(f"[geometry] OK: {cid}")

    print("[geometry] All checked cases OK.")


if __name__ == "__main__":
    main()
