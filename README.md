# Neuromorphic HAR

Code for reproducing the experiments from:

**Neuromorphic Activity Monitoring for Elderly Care Using Event-Based IMU Encoding and Spiking Neural Networks**

The paper describes the full method, including the ΣΔ event-based IMU encoding, the dual-head spiking convolutional network, and the energy estimation procedure. This repository is intended mainly to provide the code needed to reproduce the reported experiments.

## Environment

The experiments were implemented in **Python 3.10** using **Conda**.

The default environment assumes an NVIDIA GPU. In our experiments, we used:

- Python 3.10
- PyTorch
- CUDA 12.1 through `pytorch-cuda=12.1`
- NumPy
- scikit-learn
- matplotlib
- einops

Create the environment with:

```bash
conda env create -f environment.yml
conda activate neuromorphic-har
```

Verify the installation:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
python -c "import numpy, sklearn, matplotlib, einops; print('OK')"
```

The provided `environment.yml` uses `pytorch-cuda=12.1`, which corresponds to the CUDA version used in our setup. Users with a different CUDA version may need to adjust the PyTorch/CUDA dependencies according to their local system.

## Datasets

The experiments use public datasets:

- **SisFall** for fall, dynamic activity, and static-posture monitoring.
- **UCI HAR** for generic human activity recognition.

Datasets are not included in this repository. Download them from their original sources and place them in the expected data directory.

A suggested structure is:

```text
data/
├── SisFall/
└── UCI_HAR/
```

## Repository structure

```text
.
├── data/                  # Datasets, not included in the repository
├── models/                # Neural network models
│   ├── spike.py           # Spiking convolutional models and losses
│   ├── backbones.py       # Baseline/backbone models
│   ├── attention.py
│   └── MMB.py
├── scripts/               # Training, evaluation, preprocessing, energy estimation
├── results/               # Generated results, logs and metrics
├── environment.yml
└── README.md
```

## Running experiments

The exact scripts may depend on the final organization of the repository. A typical workflow is:

### 1. Preprocess data

```bash
python scripts/preprocess_sisfall.py
python scripts/preprocess_uci.py
```

### 2. Train the SisFall model

```bash
python scripts/train_sisfall.py
```

### 3. Evaluate the SisFall model

```bash
python scripts/evaluate_sisfall.py
```

### 4. Train and evaluate on UCI HAR

```bash
python scripts/train_uci.py
python scripts/evaluate_uci.py
```

### 5. Estimate inference energy

```bash
python scripts/estimate_energy.py
```

See the paper for the full experimental setup, ablations, task formulation, and energy analysis.

## Citation

If you use this code, please cite:

```bibtex
@article{valdivia2026neuromorphic,
  title   = {Neuromorphic Activity Monitoring for Elderly Care Using Event-Based IMU Encoding and Spiking Neural Networks},
  author  = {Valdivia, Sebastián and Yunge, Daniel},
  journal = {Neuromorphic Computing and Engineering},
  year    = {2026}
}
```

## License

This repository is released for research and reproducibility purposes. Please check the license file for usage terms.
