# HPE-Li

This repo started from the official implementation for [**HPE-Li: WiFi-enabled Lightweight Dual Selective Kernel Convolution for Human Pose Estimation**](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04496.pdf), published at **ECCV 2024**. It now focuses on a thesis-oriented **HPE-Li-3D** extension for WiFi-based 3D human pose estimation on MM-Fi.

See [docs/project_structure.md](docs/project_structure.md) for the current project layout and cleanup policy.

## HPE-Li Framework

![image](assets/images/Architecture.jpg)

## DSK Convolution

![image](assets/images/DSKconv.jpg)

## Data Preparation
### Download datasets.

#### Dataset

- MM-Fi Dataset

#### MM-Fi Dataset

1. Request dataset [here](https://ntu-aiot-lab.github.io/mm-fi)
2. Download the WiFi datasets:
   1. `MMFI_Dataset.zip`
   2. `MMFI_action_segments.csv`
3. Unzip all files from `MMFI_Dataset.zip` to `./data/mmfi/dataset` following directory structure:

```
- data/
  - mmfi/
    - dataset/
      - E01/
        - S01/
          - A01/
            - rgb/
            - mmwave/
            - wifi-csi/
              ...
```

#### Person-in-WiFi-3D Dataset (New Dataset)

1. Request dataset [here](https://aiotgroup.github.io/Person-in-WiFi-3D/)
2. Unzip all files from `wifipose_dataset.zip` to `./data/person_in_wifi_3d/` following directory structure:

```
- data/
  - person_in_wifi_3d/
    - train_data/
    - test_data/
```

#### Some new datasets
1. [XRF V2](https://dl.acm.org/doi/10.1145/3749521)
2. [XRF55](https://dl.acm.org/doi/10.1145/3643543)

   You guys can read [this survey](https://arxiv.org/pdf/2503.08008v2) to update more interesting things!

## Run Simulations

### MM-Fi dataset
```
python att_mmfi.py 
```

### Check Complexity

```
python comlexity.py
```
### Using Denoiser

```
python denoiser_training.py
```

## HPE-Li-3D Thesis Extension

Train the 3D baseline on MM-Fi:

```powershell
python train_3d_baseline.py --dataset-root $env:MMFI_DATASET_ROOT --split-to-use random_split --device cuda
```

Evaluate a 3D checkpoint and export both thesis metrics and body-scale g_PCK benchmark metrics:

```powershell
python tools/evaluate_3d_checkpoint.py `
  --checkpoint checkpoints/phase_b_s1_random_20e_package/best_s1_random_20e.pt `
  --dataset-root $env:MMFI_DATASET_ROOT `
  --eval-split test `
  --batch-size 8 `
  --num-workers 0 `
  --device cpu `
  --method-name "HPE-Li-3D"
```

The main 3D metrics are MPJPE, PA-MPJPE, PCK@50mm, PCK@100mm, and per-joint MPJPE. `g_PCK@10/20/30/40/50` uses thresholds based on the ground-truth R.Hip-to-L.Shoulder body scale (joint indices 1 and 11), not millimeters. This corrected definition is not directly comparable to legacy GraphPose-Fi values computed with indices 5 and 12.

## Visualization 

![image](assets/images/Visualization.jpg)

## Acknowledgements

This repo is based on [MetaFi++](https://github.com/pridy999/metafi_pose_estimation) and [SKNet](https://arxiv.org/pdf/1903.06586). The MM-Fi data processing is borrowed from [MM-Fi](https://github.com/ybhbingo/MMFi_dataset).

Thanks to the original authors for their work!


## Citation

Please cite this work if you find it useful.

```
@inproceedings{d2024hpe,
  title={Hpe-li: Wifi-enabled lightweight dual selective kernel convolution for human pose estimation},
  author={D. Gian, Toan and Dac Lai, Tien and Van Luong, Thien and Wong, Kok-Seng and Nguyen, Van-Dinh},
  booktitle={European Conference on Computer Vision},
  pages={93--111},
  year={2024},
  organization={Springer}
}
```





