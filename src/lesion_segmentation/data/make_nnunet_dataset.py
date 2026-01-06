"""
Assemble an nnU-Net–compliant dataset from preprocessed volumes.

This module organizes converted NIfTI files into the required nnU-Net
directory structure, assigns channel indices, applies train/validation
splits, and generates the dataset.json metadata file.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from lesion_segmentation.paths import load_dataset_paths
from lesion_segmentation.data.splits import load_subset_list


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset_id", type=int, default=1)
    ap.add_argument("--dataset_name", default="BONBID")
    args = ap.parse_args()

    dp = load_dataset_paths(args.config)
    case_ids = load_subset_list(dp.manifests_path)

    ds_dir = dp.nnunet_raw_root / f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    imagesTr = ds_dir / "imagesTr"
    labelsTr = ds_dir / "labelsTr"
    imagesTs = ds_dir / "imagesTs"
    imagesTr.mkdir(parents=True, exist_ok=True)
    labelsTr.mkdir(parents=True, exist_ok=True)
    imagesTs.mkdir(parents=True, exist_ok=True)

    # Copy training cases
    for cid in case_ids:
        adc = dp.converted_root / f"{cid}_ADC.nii.gz"
        lbl = dp.converted_root / f"{cid}_lesion.nii.gz"

        if not adc.exists():
            raise FileNotFoundError(f"Missing converted ADC: {adc}")
        if not lbl.exists():
            raise FileNotFoundError(f"Missing converted label: {lbl}")

        copy(adc, imagesTr / f"{cid}_0000.nii.gz")

        if dp.use_zmap:
            z = dp.converted_root / f"{cid}_Z.nii.gz"
            if not z.exists():
                raise FileNotFoundError(f"Missing converted Z: {z}")
            copy(z, imagesTr / f"{cid}_0001.nii.gz")

        copy(lbl, labelsTr / f"{cid}.nii.gz")

        print(f"[nnunet_raw] added {cid}")

    # dataset.json
    channel_names = {"0": "ADC"}
    if dp.use_zmap:
        channel_names["1"] = "Z_ADC"

    dataset_json = {
        "channel_names": channel_names,
        "labels": {"background": 0, "lesion": 1},
        "numTraining": len(case_ids),
        "file_ending": ".nii.gz",
    }

    with (ds_dir / "dataset.json").open("w") as f:
        json.dump(dataset_json, f, indent=2)

    print(f"[nnunet_raw] Wrote dataset.json and data to: {ds_dir}")


if __name__ == "__main__":
    main()
