# Lesion_segmentation
Lesion_segmentation/
  README.md
  .gitignore
  pyproject.toml        (or requirements.txt)

  src/
    lesion_segmentation/
      __init__.py
      config.py
      paths.py
      data/
        dataset_index.py
        convert_mha_to_nifti.py
        make_nnunet_dataset.py
        splits.py
      training/
        train_nnunet.py
        predict_nnunet.py
      evaluation/
        metrics.py

  utils/
    sanity_check_geometry.py
    inspect_case.py

  configs/
    dataset_boston.yaml
    nnunet.yaml

  datasets/             (local only; ignored)
    Boston/
      original/
        BONBID2023_Train/...
        BONBID2023_Val/...
        BONBID2023_Test.zip ...
        atlases/...
      nnunet_raw/
        Dataset001_BONBID/
          imagesTr/
          labelsTr/
          imagesTs/
          dataset.json
      splits/
        split_v1.json
      subsets/
        small_debug.txt
        train_50.txt
