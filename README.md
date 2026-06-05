# Neuromorphic HAR

Event-based IMU encoding and spiking neural network training for SisFall and UCI HAR.

The repository provides scripts to:

- generate Sigma-Delta event tensors,
- autotag SisFall static-posture windows,
- optimize Sigma-Delta theta values,
- train SisFall dual-head or single-head SCN models,
- train UCI HAR single-head SCN models,
- estimate normalized inference-stage energy consumption.

## Environment

Recommended:

- Python 3.10 or 3.11
- PyTorch 2.x
- CUDA GPU for full training runs; CPU is enough for preprocessing and small tests

Virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Conda:

```bash
conda env create -f environment.yml
conda activate neuromorphic-har
```

For CUDA, install the PyTorch build that matches your driver and platform.

# Data layout

Datasets are not included in this repository. Download them manually and place them under `data/`.

```text
data/
  SisFall_dataset/
  labels_transitions/
  UCI HAR Dataset/
```

## SisFall

Download:

```text
https://www.kaggle.com/datasets/nvnikhil0001/sis-fall-original-dataset
```

Place the extracted subject folders inside:

```text
data/SisFall_dataset/
```

Reference:

```bibtex
@article{sucerquia2017sisfall,
  title={SisFall: A Fall and Movement Dataset},
  author={Sucerquia, Angela and L{\'o}pez, Jos{\'e} David and Vargas-Bonilla, Jes{\'u}s Francisco},
  journal={Sensors},
  volume={17},
  number={1},
  pages={198},
  year={2017},
  doi={10.3390/s17010198}
}
```

## UCI HAR

Download:

```text
https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
```

Place the extracted dataset contents inside:

```text
data/UCI HAR Dataset/
```

Reference:

```bibtex
@inproceedings{anguita2013public,
  title={A Public Domain Dataset for Human Activity Recognition Using Smartphones},
  author={Anguita, Davide and Ghio, Alessandro and Oneto, Luca and Parra, Xavier and Reyes-Ortiz, Jorge L.},
  booktitle={Proceedings of the 21st European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning},
  pages={437--442},
  year={2013}
}
```

## Help

All scripts expose their options through argparse:

```bash
python run_posture_tagging.py --help
python optimize_theta.py --help
python train_sisfall.py --help
python train_ucihar.py --help
python energy_report.py --help
```

## SisFall autotagging

```bash
python posture_tagging.py 
```

Useful parser options include `--refresh`, `--save-plot`, `--th-pt`, `--th-sp`, and `--margin`.

## Theta optimization

SisFall:

```bash
python optimize_theta.py --dataset sisfall --data-root data --ups 5 --dead-zone 0.5 --quiet-epochs
```

UCI HAR:

```bash
python optimize_theta.py --dataset ucihar --data-root data --ups 5 --dead-zone 0.5 --quiet-epochs
```

Fast debug run:

```bash
python optimize_theta.py --dataset sisfall --data-root data --ups 5 --dead-zone 0.5 \
  --train-per-class 16 --val-per-class 8 --test-per-class 8 \
  --sampled-test-only --max-epochs 3 --workers 1 --executor none --force
```

## Training

SisFall dual-head:

```bash
python train_sisfall.py --head-mode dual --epochs 100 --batch-size 512 --workers 4
```

SisFall single-head:

```bash
python train_sisfall.py --head-mode single --epochs 100 --batch-size 512 --workers 4
```

UCI HAR:

```bash
python train_ucihar.py --epochs 100 --batch-size 256 --workers 4
```

Example parameter entry:

```bash
python train_sisfall.py --head-mode dual --tau 0.75 --thresh 0.5 \
  --conv-ch 32 64 64 --kernels 32 32 8 --strides 4 2 1 \
  --p-drop 0.35 --dead-zone 0.5 --run sisfall_custom
```

Each training run creates a separate folder under `runs/`.

## Energy reports

All trained models for one dataset:

```bash
python energy_report.py --dataset ucihar 
python energy_report.py --dataset sisfall
```

Single run:

```bash
python energy_report.py --dataset sisfall --run-dir runs/sisfall_scn_YYYYMMDD_HHMMSS
```

The estimator reports normalized network-only inference energy. It counts dense-equivalent MACs, effective SOPs from measured sparsity, LIF updates, and optional scratchpad data movement. The analog event-generation front-end is not included.

## Reproducibility

Exact results are not guaranteed across runs. Differences can appear due to random seeds, hardware, CUDA/cuDNN behavior, library versions, dataset split choices, cache state, and early stopping.

## References and related code

This project builds on prior work in spiking neural networks for human activity recognition and sparsity-aware energy estimation. The SNN-HAR repository was used as a reference for the general spiking HAR modeling and training workflow, while SATA/SATA_Sim was used as a reference for normalized operation-level energy accounting in sparse SNN inference.

- Li Y, Yin R, Kim Y, Panda P. *Efficient human activity recognition with spatio-temporal spiking neural networks*. Frontiers in Neuroscience, 2023. Code: https://github.com/Intelligent-Computing-Lab-Panda/SNN_HAR
- Yin R, Moitra A, Bhattacharjee A, Kim Y, Panda P. *SATA: Sparsity-Aware Training Accelerator for Spiking Neural Networks*. IEEE TCAD, 2023. Code: https://github.com/RuokaiYin/SATA_Sim