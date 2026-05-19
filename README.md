# Neuromorphic Activity Monitoring for Elderly Care

Implementation of a neuromorphic Human Activity Recognition (HAR) pipeline for elderly-care monitoring using event-based IMU encoding and Spiking Neural Networks (SNNs).

This repository accompanies the paper:

> **Neuromorphic Activity Monitoring for Elderly Care Using Event-Based IMU Encoding and Spiking Neural Networks**  
> Sebastián Valdivia and Daniel Yunge  
> School of Electrical Engineering, Pontificia Universidad Católica de Valparaíso, Chile

The system converts six-channel inertial windows from a wearable IMU into sparse positive/negative events using a ΣΔ encoder. A dual-head spiking convolutional network then classifies global activity contexts and refines static-posture states.

---

## Overview

Continuous monitoring for Ambient-Assisted Living (AAL) requires models that are accurate, low-power, and suitable for wearable deployment. This project explores a neuromorphic approach where both sensing and inference are sparse:

1. A six-axis IMU provides tri-axial accelerometer and gyroscope signals.
2. A ΣΔ event encoder converts dense inertial signals into sparse binary events.
3. A low-rate offset vector preserves the absolute inertial reference of each window.
4. A spiking convolutional network performs activity recognition using event-driven inference.
5. A dual-head formulation separates global activity recognition from static-posture refinement.

<p align="center">
  <img src="assets/pipeline_overview.png" width="750" alt="Neuromorphic IMU activity monitoring pipeline">
</p>

---

## Main Contributions

- **Event-based IMU representation** based on ΣΔ encoding of accelerometer and gyroscope windows.
- **Auxiliary offset input** to preserve absolute inertial information not retained by the event stream.
- **Dual-head SNN classifier** for elderly-care activity monitoring:
  - Head 1: `Fall`, `Dynamic`, `Static`
  - Head 2: `Stable Posture`, `Postural Transition`
- **Transition-aware static-label refinement** for separating stable postures from posture transitions.
- **Energy-aware evaluation** using normalized operation-level estimates for sparse spiking inference.
- **Cross-dataset validation** on SisFall and UCI HAR.

---

## Method

### 1. Input Signal

Each sample window contains six IMU channels:

```text
[acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
```

For SisFall, the original recordings are sampled at `200 Hz`. Windows are extracted with:

```text
window length = 410 samples
window duration = 2.05 s
overlap = 50%
```

Fall windows are peak-centered around the maximum absolute acceleration peak.

---

### 2. Event-Based IMU Encoding

The dense IMU signal is converted into sparse events with a channel-wise ΣΔ encoder.

For each channel, the encoder tracks a reconstruction state and emits an event only when the signal deviation crosses a threshold:

```text
+1 event: signal increases above threshold
-1 event: signal decreases below threshold
 0 event: no significant change
```

Positive and negative polarities are stored separately, producing an event tensor:

```text
E ∈ {0, 1}^{Te × C × 2}
```

where:

```text
Te = number of event time steps
C  = 6 IMU channels
2  = positive and negative polarities
```

For SisFall, the selected configuration uses:

```text
upsampling factor = ×5
Te = 2048
dead-zone factor = 0.5
```

An auxiliary offset vector is also stored:

```text
x0 ∈ R^6
```

This vector corresponds to the first sample of the window and provides the absolute inertial reference required for posture-related discrimination.

---

### 3. Spiking Neural Network

The classifier is a dual-head spiking convolutional network.

The event branch contains three spiking convolutional blocks:

```text
Conv1D → BatchNorm → LIF → MaxPool
```

After the convolutional backbone, temporal activity is collapsed using firing-rate features.

The offset branch processes the offset vector with a fully connected LIF layer. Both branches are concatenated before the readout heads.

```text
Event tensor ──► Spiking Conv Blocks ──► Firing-rate features ┐
                                                               ├──► Head 1: Fall / Dynamic / Static
Offset vector ─► LIF offset branch ────────────────────────────┘
                                                               └──► Head 2: Stable / Transition
```

---

## Selected Model

The selected operating point, `SCNsel`, balances accuracy, parameter count, and estimated energy.

| Parameter | Value |
|---|---:|
| Channels | `(32, 64, 64)` |
| Kernel sizes | `(32, 32, 8)` |
| Strides | `(4, 2, 1)` |
| Polarity handling | Separate positive/negative channels |
| Parameters | `111,276` |
| Head 1 accuracy | `94.27%` |
| Head 2 accuracy | `89.88%` |
| Mean accuracy | `92.08%` |
| EPE | `4.214 × 10^6` |
| Compute reduction vs. dense | `97.74%` |

---

## Results

### SisFall

| Model | Head 1 Acc. | Head 2 Acc. | Mean Acc. | Params | EPE |
|---|---:|---:|---:|---:|---:|
| `SCNacc` | `94.45%` | `89.93%` | `92.19%` | `144,740` | `7.864 × 10^6` |
| `SCNsel` | `94.27%` | `89.88%` | `92.08%` | `111,276` | `4.214 × 10^6` |
| `SCNminE` | `93.20%` | `88.30%` | `90.75%` | `55,980` | `0.381 × 10^6` |

### UCI HAR

The same event-based representation and selected SCN architecture were also evaluated on UCI HAR.

| Dataset | Classes | Accuracy |
|---|---:|---:|
| UCI HAR | 6 | `96.12%` |

---

## Repository Structure

```text
Neuromorphic-HAR/
├── assets/                  # Figures used in the README and paper
├── configs/                 # YAML configuration files
│   ├── sisfall_scnsel.yaml
│   └── ucihar_scnsel.yaml
├── data/                    # Dataset root directory, not tracked by Git
│   ├── SisFall/
│   └── UCI_HAR/
├── notebooks/               # Exploratory analysis and visualizations
├── scripts/                 # Reproducible experiment entry points
│   ├── prepare_sisfall.py
│   ├── prepare_ucihar.py
│   ├── train.py
│   ├── evaluate.py
│   └── estimate_energy.py
├── src/
│   ├── datasets/            # Dataset loaders and windowing logic
│   ├── encoding/            # ΣΔ event encoder and threshold optimization
│   ├── models/              # Spiking convolutional networks
│   ├── training/            # Losses, metrics, and training loops
│   ├── energy/              # Operation-level energy estimation
│   └── utils/               # Shared utilities
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Installation

```bash
git clone https://github.com/s-valdivia-vasquez/Neuromorphic-HAR.git
cd Neuromorphic-HAR

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

For GPU support, install the PyTorch version compatible with your CUDA version before installing the remaining dependencies.

---

## Datasets

This repository uses public HAR datasets. They are not included in the repository.

### SisFall

Download SisFall from its original source and place it under:

```text
data/SisFall/
```

Expected task grouping:

| Group | Sessions |
|---|---|
| Fall | `F01–F15` |
| Dynamic | `D01–D06`, `D18–D19` |
| Static | `D07–D17` |

### UCI HAR

Download UCI HAR from its original source and place it under:

```text
data/UCI_HAR/
```

---

## Usage

### Prepare SisFall

```bash
python scripts/prepare_sisfall.py \
  --data-root data/SisFall \
  --output-dir processed/sisfall \
  --window-size 410 \
  --overlap 0.5
```

### Optimize ΣΔ Thresholds

```bash
python scripts/optimize_thresholds.py \
  --config configs/sisfall_scnsel.yaml \
  --data-dir processed/sisfall
```

### Train SCN on SisFall

```bash
python scripts/train.py \
  --config configs/sisfall_scnsel.yaml
```

### Evaluate SisFall Model

```bash
python scripts/evaluate.py \
  --config configs/sisfall_scnsel.yaml \
  --checkpoint checkpoints/sisfall_scnsel.pt
```

### Estimate Energy

```bash
python scripts/estimate_energy.py \
  --config configs/sisfall_scnsel.yaml \
  --checkpoint checkpoints/sisfall_scnsel.pt
```

### Train on UCI HAR

```bash
python scripts/train.py \
  --config configs/ucihar_scnsel.yaml
```

---

## Configuration Example

```yaml
encoder:
  type: sigma_delta
  upsampling_factor: 5
  dead_zone: 0.5
  polarities: separate

windowing:
  window_size: 410
  overlap: 0.5

model:
  name: scn_dual_head
  channels: [32, 64, 64]
  kernels: [32, 32, 8]
  strides: [4, 2, 1]
  offset_hidden: 8
  lif:
    tau_mem: 0.75
    v_th: 0.5
    reset: soft

training:
  optimizer: adam
  batch_size: 64
  gradient_clip: true
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
```

---

## Citation

Please cite the associated paper if you use this repository:

```bibtex
@misc{valdivia_yunge_neuromorphic_activity_monitoring,
  title  = {Neuromorphic Activity Monitoring for Elderly Care Using Event-Based IMU Encoding and Spiking Neural Networks},
  author = {Valdivia, Sebasti{\'a}n and Yunge, Daniel},
  note   = {Manuscript},
  year   = {YYYY}
}
```

---

## License

This project is released under the license specified in [`LICENSE`](LICENSE).

---

## Acknowledgements

This work was supported by the Agencia Nacional de Investigación y Desarrollo (ANID), Chile, under FONDECYT Initiation Grant No. 11251536, Project LANTERN: Low-Power Adaptive Neuromorphic System for Assisted Living.
