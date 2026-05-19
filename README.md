# Neuromorphic HAR

Code repository for the paper **“Neuromorphic Activity Monitoring for Elderly Care Using Event-Based IMU Encoding and Spiking Neural Networks”**.

This repository provides the Python code needed to reproduce the experiments reported in the paper: event-based IMU encoding, spiking neural network training, evaluation, and energy estimation.

## Requirements

The project was developed with:

- Python 3.10
- Conda
- PyTorch
- NumPy

A complete Conda environment file is provided:

```bash
conda env create -f environment.yml
conda activate neuromorphic-har
```

## Repository structure

```text
.
├── configs/              # Experiment configuration files
├── data/                 # Dataset location; not tracked by git
├── scripts/              # Training, evaluation and preprocessing scripts
├── src/                  # Core implementation
│   ├── encoding/         # Sigma-delta event-based IMU encoder
│   ├── models/           # Spiking convolutional networks
│   ├── training/         # Training and validation routines
│   ├── evaluation/       # Metrics and reports
│   └── energy/           # Energy estimation utilities
├── environment.yml       # Conda environment
└── README.md
```

## Datasets

The experiments use public datasets:

- **SisFall** for elderly-care activity monitoring, fall detection, dynamic activity recognition, and static-posture refinement.
- **UCI HAR** for additional validation on a generic HAR benchmark.

Place the datasets inside `data/` following the expected structure used by the preprocessing scripts.

```text
data/
├── sisfall/
└── uci_har/
```

The datasets are not included in this repository.

## Usage

### 1. Preprocess data

```bash
python scripts/preprocess_sisfall.py --config configs/sisfall.yaml
python scripts/preprocess_uci_har.py --config configs/uci_har.yaml
```

### 2. Train the model

```bash
python scripts/train.py --config configs/sisfall_scnsel.yaml
```

### 3. Evaluate

```bash
python scripts/evaluate.py --config configs/sisfall_scnsel.yaml --checkpoint checkpoints/best.pt
```

### 4. Estimate energy

```bash
python scripts/estimate_energy.py --config configs/sisfall_scnsel.yaml --checkpoint checkpoints/best.pt
```

## Main components

- Sigma-delta event-based encoder for six-channel IMU windows.
- Auxiliary offset vector for preserving the initial inertial reference.
- Dual-head spiking convolutional network for:
  - global classification: Fall / Dynamic / Static
  - static refinement: Stable posture / Postural transition
- Energy estimation based on sparse operations and LIF activity.

## Reproducing paper results

The selected operating point in the paper corresponds to the `SCNsel` configuration. Use:

```bash
python scripts/train.py --config configs/sisfall_scnsel.yaml
```

Results may vary slightly depending on hardware, random seed, library versions, and dataset preprocessing.

## Citation

```bibtex
@article{valdivia2026neuromorphic,
  title   = {Neuromorphic Activity Monitoring for Elderly Care Using Event-Based IMU Encoding and Spiking Neural Networks},
  author  = {Valdivia, Sebasti\'an and Yunge, Daniel},
  journal = {Neuromorphic Computing and Engineering},
  year    = {2026}
}
```

## License

Add the license selected for this repository.
