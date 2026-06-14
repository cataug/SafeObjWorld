# SafeObjWorld

<p align="center">
  <b>Auditing Object Permanence, Physical Hallucinations, and Stress Robustness in Video World-Model Rollouts</b>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#method">Method</a> •
  <a href="#experiments">Experiments</a> •
  <a href="#results">Results</a> •
  <a href="#figures">Figures</a> •
  <a href="#reproduce">Reproduce</a>
</p>

---

## Overview

**SafeObjWorld** is an experimental framework for evaluating whether video world models preserve **object permanence** and avoid physically implausible rollouts under visual and temporal stress.

Instead of treating future-mask prediction only as an overlap problem, SafeObjWorld asks a safety-oriented question:

> Can a model keep tracking an object as a persistent physical entity when visibility degrades, frames are dropped, or motion becomes uncertain?

The framework evaluates future object-mask rollouts on DAVIS-style video object segmentation data and compares naive propagation, optical-flow propagation, ConvLSTM rollouts, object-token rollouts, and the proposed reliability-regularized **SafeObjWorld-R** variants.

---

## Why This Matters

Standard overlap metrics such as IoU and J&F can hide physically implausible behavior.

A model may achieve reasonable mask overlap while still producing unsafe rollout failures such as:

- object disappearance without evidence;
- hallucinated object regions;
- fragmented object masks;
- unstable motion trajectories;
- poor uncertainty under stress.

SafeObjWorld exposes these failures using object-centric reliability metrics and stress tests.

---

## Core Idea

SafeObjWorld evaluates future object rollouts using both standard segmentation quality and physical reliability.

Given a short input prefix:

```text
RGB frames + object masks at t-2, t-1, t
````

the model predicts object masks at future steps:

```text
t+1, t+2, ..., t+H
```

The prediction is evaluated not only by mask overlap, but also by whether the predicted object remains physically coherent.

---

## Method

### SafeObjWorld-R

The main method is **SafeObjWorld-R**, a reliability-regularized future object-mask world model.

It uses a ConvLSTM-style rollout backbone and adds object-level reliability constraints:

```text
SafeObjWorld-R =
    ConvLSTM future-mask rollout
  + stress training
  + temporal consistency loss
  + split proxy loss
  + area stability loss
  + stochastic MC-dropout uncertainty
```

### Main Reliability Targets

SafeObjWorld-R is designed to reduce:

| Failure type              | Meaning                                                |
| ------------------------- | ------------------------------------------------------ |
| Vanishing                 | object disappears despite being present                |
| Hallucination             | object appears without support                         |
| Splitting                 | one object becomes fragmented into multiple components |
| Motion inconsistency      | predicted object trajectory becomes unstable           |
| Miscalibrated uncertainty | model confidence does not match rollout error          |

---

## Repository Structure

```text
SafeObjWorld/
├── code/
│   ├── safeobjworld_v2.py
│   └── launch_safeobjworld_v2.py
│
├── runs_v2_full/
│   ├── RESULTS/
│   │   ├── all_summaries.csv
│   │   ├── all_rows.csv
│   │   ├── table_1_main_all_methods.csv
│   │   ├── table_2_clean_reliability.csv
│   │   ├── table_3_stress_drop.csv
│   │   ├── table_4_horizon_breakdown.csv
│   │   ├── EXP1_standard_prediction_clean.csv
│   │   ├── EXP2_reliability_audit_all_stress.csv
│   │   ├── EXP3_stress_robustness.csv
│   │   ├── EXP5_uncertainty_calibration.csv
│   │   └── EXP6_ablation_study.csv
│   │
│   └── logs/
│
├── figures_all_results/
│   ├── fig01_clean_leaderboard_H5.png
│   ├── fig02_jf_vs_split_H5_occlusion.png
│   ├── fig03_clean_to_occlusion_arrows_H5.png
│   ├── fig05_heatmap_jf_H5.png
│   ├── fig08_ablation_dashboard_H5_occlusion.png
│   └── fig10_radar_occlusion_H5.png
│
├── figures_qualitative/
│   └── qualitative rollout visualizations
│
├── run_v2_full.sh
├── setup_safeobjworld_v2.sh
└── git_project_manifest.txt
```

Large datasets, virtual environments, checkpoints, model weights, caches, and archives are intentionally excluded from Git.

---

## Experiments

SafeObjWorld evaluates models under multiple rollout horizons and stress conditions.

### Prediction Horizons

```text
H = 5
H = 10
```

Per-step metrics are also reported for each future step:

```text
t+1, t+2, ..., t+H
```

### Stress Conditions

| Stress     | Purpose                                     |
| ---------- | ------------------------------------------- |
| Clean      | standard future rollout                     |
| Occlusion  | object permanence under partial visibility  |
| Blur       | degraded visual evidence                    |
| Frame drop | missing temporal evidence                   |
| Low light  | visual degradation                          |
| Noise      | sensor-like corruption                      |
| Distractor | robustness to confusing object-like regions |

---

## Compared Methods

### Naive Baselines

| Method          | Description                             |
| --------------- | --------------------------------------- |
| Copy-last       | repeats the last observed object mask   |
| Linear centroid | extrapolates mask using centroid motion |
| Linear + scale  | centroid motion with area extrapolation |

### Classical Motion Baselines

| Method          | Description                                          |
| --------------- | ---------------------------------------------------- |
| Flow-warp       | propagates the mask using optical flow               |
| Flow + fallback | optical flow with linear fallback at longer horizons |

### Learned Baselines

| Method            | Description                               |
| ----------------- | ----------------------------------------- |
| ConvLSTM          | future-mask rollout baseline              |
| ConvLSTM + stress | ConvLSTM trained with stress augmentation |

### SafeObjWorld-R Variants

| Method         | Description                        |
| -------------- | ---------------------------------- |
| SafeObjWorld-R | full reliability-regularized model |
| w/o temporal   | removes temporal consistency       |
| w/o split      | removes split proxy loss           |
| w/o area       | removes area stability loss        |
| w/o stress     | removes stress training            |
| deterministic  | removes stochastic uncertainty     |
| image-only     | removes object mask conditioning   |
| mask-only      | removes RGB appearance             |

### Object-Token Ablations

| Method                | Description                                    |
| --------------------- | ---------------------------------------------- |
| Object-token          | compact object-state rollout                   |
| Object-token det.     | deterministic object-token rollout             |
| Object-token w/o geom | object-token rollout without geometry features |

---

## Metrics

### Standard Mask Quality

| Metric     | Direction | Description                  |
| ---------- | --------: | ---------------------------- |
| IoU        |         ↑ | mask overlap                 |
| Boundary-F |         ↑ | boundary agreement           |
| J&F        |         ↑ | combined DAVIS-style quality |

### Object-Centric Reliability

| Metric        | Direction | Description                   |
| ------------- | --------: | ----------------------------- |
| OPS           |         ↑ | object permanence score       |
| Vanish        |         ↓ | disappearance rate            |
| Hallucination |         ↓ | unsupported object appearance |
| Split         |         ↓ | fragmented object prediction  |
| MCE           |         ↓ | motion consistency error      |

### Uncertainty and Calibration

| Metric            | Direction | Description                                   |
| ----------------- | --------: | --------------------------------------------- |
| UEC               |         ↑ | uncertainty-error correlation                 |
| ECE               |         ↓ | expected calibration error                    |
| AURC              |         ↓ | area under risk-coverage curve                |
| Selective J&F@80% |         ↑ | quality after rejecting uncertain predictions |

---

## Results

### Key Finding 1

**Overlap metrics alone are not enough.**

Flow-based methods can achieve competitive J&F, but often produce fragmented object predictions. SafeObjWorld explicitly exposes this gap by reporting Split Rate, OPS, Vanish Rate, Hallucination Rate, and MCE.

### Key Finding 2

**Stress training is crucial for object permanence.**

The clean-only ablation performs well on clean sequences but degrades strongly under occlusion. This shows that robustness cannot be inferred from clean validation alone.

### Key Finding 3

**SafeObjWorld-R improves the reliability-quality trade-off.**

SafeObjWorld-R preserves competitive future-mask quality while reducing physical failure modes under stress.

Example H=5 occlusion trend:

| Method              |                   J&F ↑ | OPS ↑ | Split ↓ |
| ------------------- | ----------------------: | ----: | ------: |
| Flow-warp           |                   lower | lower |    high |
| ConvLSTM            |                  medium |  high |    high |
| ConvLSTM + stress   |                  higher |  high |   lower |
| SafeObjWorld-R      |                    high |  high |   lower |
| SafeObjWorld-R det. | highest overlap variant |  high |     low |

The deterministic variant is strong in overlap, while the full stochastic variant enables uncertainty-aware analysis.

---

## Figures

### Clean Future-Mask Quality

```text
figures_all_results/fig01_clean_leaderboard_H5.png
figures_all_results/fig01_clean_leaderboard_H10.png
```

### Quality vs Physical Splitting

```text
figures_all_results/fig02_jf_vs_split_H5_occlusion.png
figures_all_results/fig02_jf_vs_split_H10_occlusion.png
```

### Clean-to-Occlusion Shift

```text
figures_all_results/fig03_clean_to_occlusion_arrows_H5.png
figures_all_results/fig03_clean_to_occlusion_arrows_H10.png
```

### Stress Robustness Heatmaps

```text
figures_all_results/fig05_heatmap_jf_H5.png
figures_all_results/fig05_heatmap_ops_H5.png
figures_all_results/fig05_heatmap_split_H5.png
```

### Ablation Dashboard

```text
figures_all_results/fig08_ablation_dashboard_H5_occlusion.png
```

### Reliability Trade-off Radar

```text
figures_all_results/fig10_radar_occlusion_H5.png
```

### Qualitative Rollouts

```text
figures_qualitative/
```

Qualitative figures show:

```text
input frames → tracked object mask → future ground truth → model predictions → uncertainty → reliability verdict
```

---

## Reproduce

### 1. Prepare Environment

```bash
cd ~/SafeObjWorld
SKIP_ENV=0 ./setup_safeobjworld_v2.sh
```

### 2. Smoke Test

```bash
cd ~/SafeObjWorld
KILL_OLD=1 CLEAN=1 QUICK=1 ./run_v2_full.sh
```

Smoke-test outputs are written to:

```text
runs_v2_smoke/
```

### 3. Full Experiment

```bash
cd ~/SafeObjWorld
KILL_OLD=1 CLEAN=1 QUICK=0 ./run_v2_full.sh
```

Full outputs are written to:

```text
runs_v2_full/
```

### 4. Full Experiment with H=10 and Optional Stress Tests

```bash
cd ~/SafeObjWorld
KILL_OLD=1 CLEAN=1 QUICK=0 RUN_H10=1 OPTIONAL_STRESS=1 ./run_v2_full.sh
```

### 5. Monitor Logs

```bash
tail -f ~/SafeObjWorld/runs_v2_full/logs/*.log
```

### 6. Monitor GPU

```bash
watch -n 2 nvidia-smi
```

---

## Result Files

Main aggregated outputs:

```text
runs_v2_full/RESULTS/all_summaries.csv
runs_v2_full/RESULTS/all_rows.csv
runs_v2_full/RESULTS/table_1_main_all_methods.csv
runs_v2_full/RESULTS/table_2_clean_reliability.csv
runs_v2_full/RESULTS/table_3_stress_drop.csv
runs_v2_full/RESULTS/table_4_horizon_breakdown.csv
```

Experiment-specific tables:

```text
runs_v2_full/RESULTS/EXP1_standard_prediction_clean.csv
runs_v2_full/RESULTS/EXP2_reliability_audit_all_stress.csv
runs_v2_full/RESULTS/EXP3_stress_robustness.csv
runs_v2_full/RESULTS/EXP5_uncertainty_calibration.csv
runs_v2_full/RESULTS/EXP6_ablation_study.csv
```

---

## Plotting

### Metric Dashboards

Use the notebook/code that generates:

```text
figures_all_results/
```

It creates:

* clean leaderboard plots;
* J&F vs Split quadrant plots;
* stress heatmaps;
* horizon degradation curves;
* ablation dashboards;
* uncertainty/calibration plots;
* radar plots.

### Qualitative Rollout Figures

Use the qualitative plotting notebook/code to generate:

```text
figures_qualitative/
```

These figures visualize:

```text
input prefix frames
tracked object masks
future ground truth
baseline predictions
SafeObjWorld-R predictions
uncertainty maps
reliability summaries
```

---

## Main Claims

SafeObjWorld supports three core claims:

1. **Standard future-mask overlap metrics hide physical failures.**
   Methods with competitive J&F may still split, vanish, or hallucinate objects.

2. **Stress testing reveals failures not visible on clean data.**
   Clean-only performance is insufficient for evaluating video world-model reliability.

3. **Reliability regularization improves the robustness-quality trade-off.**
   SafeObjWorld-R keeps competitive J&F while reducing physically implausible rollouts under stress.

---

## Intended Use

SafeObjWorld is useful for:

* evaluating video world models;
* auditing object permanence;
* testing embodied-AI perception rollouts;
* studying physical hallucinations;
* comparing deterministic and stochastic future prediction;
* analyzing uncertainty under degraded visual evidence.

---

## Excluded Files

The repository intentionally excludes:

```text
data/
datasets/
DAVIS raw data
virtual environments
model checkpoints
trained weights
HuggingFace caches
large archives
large numpy arrays
files larger than 95 MB
```

This keeps the repository lightweight and reproducible without storing raw datasets or heavy model artifacts.

---

## Citation

If you use SafeObjWorld, please cite this repository or the corresponding paper when available.

```bibtex
@misc{safeobjworld2026,
  title        = {SafeObjWorld: Auditing Object Permanence and Physical Hallucinations in Video World Models},
  author       = {cataug},
  year         = {2026},
  howpublished = {\url{https://github.com/cataug/SafeObjWorld}},
  note         = {Experimental framework for reliability-aware future object-mask rollouts}
}
```

---

## Project Status

SafeObjWorld is an experimental research repository.

Current contents include:

* complete experiment runner;
* dynamic GPU scheduler;
* baseline and ablation implementations;
* stress-testing protocol;
* aggregated result tables;
* logs;
* paper-style figures;
* qualitative rollout visualizations.

