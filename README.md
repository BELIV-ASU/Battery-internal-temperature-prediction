# Battery Internal Temperature Prediction
## LSTM + Physics-Informed Neural Network Framework

> **Real-time internal core temperature estimation for large-format prismatic NMC lithium-ion batteries using deep learning and physics-informed constraints.**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [The Problem We Are Solving](#2-the-problem-we-are-solving)
3. [Key Results](#3-key-results)
4. [Cell Specifications](#4-cell-specifications)
5. [Dataset](#5-dataset)
6. [Model Architecture](#6-model-architecture)
7. [Repository Structure](#7-repository-structure)
8. [Getting Started](#8-getting-started)
9. [Step-by-Step Pipeline](#9-step-by-step-pipeline)
10. [Understanding the Output](#10-understanding-the-output)
11. [Physics Model](#11-physics-model)
12. [Comparison Against Published Work](#12-comparison-against-published-work)
13. [Known Limitations](#13-known-limitations)
14. [For New Lab Members](#14-for-new-lab-members)

---

## 1. Project Overview

This project builds a machine learning framework to predict the **internal core temperature** of a lithium-ion battery cell in real time, every second, using only the sensor measurements that a Battery Management System (BMS) already has — current, voltage, surface temperature, and ambient temperature.

**Why this matters:** The temperature sensor in a real battery sits on the outside surface. During fast charging or aggressive driving, the core can be significantly hotter than the surface. By the time the surface sensor detects danger, the inside may already be in a critical state. This framework fills that thermal blind spot.

**What we built:**
- A **Baseline LSTM** model that learns purely from data
- An **LSTM + PINN** model that also enforces the laws of heat transfer during training
- A **GRU** model for architectural comparison
- An independent **MATLAB LSTM** implementation for cross-framework validation

All three models are trained and validated on **real experimental data** from the CALB L148N58A 58Ah NMC prismatic cell.

---

## 2. The Problem We Are Solving

```
WITHOUT our model:
  Surface sensor reads:  38°C
  BMS thinks:            Safe
  Reality:               Core is at 44°C
  Result:                No warning given

WITH our model:
  Surface sensor reads:  38°C
  Our model predicts:    Core is at 44°C
  BMS detects:           Approaching danger zone
  Result:                Protective action taken in time
```

For the CALB L148N58A cell which is **27mm thick**, heat generated at the core must travel **13.5mm** through dense electrode material before reaching the surface sensor. This journey takes time and creates a persistent temperature gradient that surface sensors cannot see.

---

## 3. Key Results

### Test Set Performance (6.83M real experimental measurements)

| Model | RMSE (°C) | MAE (°C) | Parameters | Physics Constraint |
|---|---|---|---|---|
| **Baseline LSTM** | **0.099** | **0.074** | 819,329 | No |
| GRU | 0.114 | 0.091 | ~615,000 | No |
| LSTM + PINN | 0.116 | 0.089 | 819,329 | Yes — two-node thermal |
| MATLAB LSTM | ~0.10 | ~0.08 | 819,329 | No (cross-validation) |

### Comparison Against Published Work

| Method | Cell Type | RMSE (°C) | vs Our Best |
|---|---|---|---|
| Karnehm et al. 2024 LSTM | Cylindrical 21700 | 0.854 | 8.6x worse |
| Karnehm et al. 2024 KAN | Cylindrical 21700 | 0.751 | 7.6x worse |
| Shen et al. 2025 PINN | Blade LFP | 0.570 | 5.8x worse |
| **Our Baseline LSTM** | **Prismatic NMC 58Ah** | **0.099** | — |

### Surface Validation (proves physics model is correct)

| Metric | Value |
|---|---|
| Mean surface RMSE | 0.187°C |
| Median surface RMSE | 0.154°C |
| Max surface RMSE | 0.564°C |
| Runs validated under 1.5°C | 100% (164/164 runs) |

---

## 4. Cell Specifications

| Property | Value |
|---|---|
| Manufacturer | CALB (China Aviation Lithium Battery) |
| Model | L148N58A |
| Chemistry | NMC (Nickel Manganese Cobalt) |
| Nominal capacity | 58 Ah |
| Nominal voltage | 3.7 V |
| Dimensions | 148 × 27 × 106 mm |
| Cell thickness | 27 mm |
| Mass | 0.93 kg |
| Max continuous temperature | 45°C |
| Application | EV powertrains and energy storage |

---

## 5. Dataset

### Source
**Mendeley Data — University of Trieste, Italy**
- URL: https://data.mendeley.com/datasets/ycx459r5c3/2
- Citation: Include in your paper when using this data

### Structure
```
battery_data/
    Temperature_10C/
        C20_charge/       (11 cells × 1 run = 11 .mat files)
        C20_discharge/    (11 .mat files)
        DV_UDDS/          (11 .mat files)
        DV_US06/          (11 .mat files)
        DV_WLTP/          (11 .mat files)
    Temperature_25C/      (same 5 folders × 11 cells)
    Temperature_40C/      (same 5 folders × 11 cells)
```
**Total: 3 temperatures × 5 profiles × 11 cells = 165 .mat files**

### Dataset Statistics

| Statistic | Value |
|---|---|
| Total rows | 6,834,132 |
| Total experimental runs | 164 |
| Number of cells tested | 11 |
| Time resolution | 1 second (after resampling) |
| T_internal range | 6.84°C to 42.36°C |
| T_internal mean | 25.76°C |
| Training data | 4,761,351 rows (114 runs, 70%) |
| Validation data | 1,034,293 rows (25 runs, 15%) |
| Test data | 1,038,488 rows (25 runs, 15%) |
| Training sequences | 475,517 (60-second sliding windows) |

### Drive Cycle Profiles

| Profile | What It Simulates | Duration |
|---|---|---|
| WLTP | Mixed city and motorway driving | ~90 minutes |
| UDDS | Urban city driving, stop-go traffic | ~70 minutes |
| US06 | Aggressive highway, fast acceleration | ~70 minutes |
| C20 charge | Very slow gentle charging | ~20 hours |
| C20 discharge | Very slow gentle discharging | ~20 hours |

### Important Note on T_internal
The CALB dataset does **not** contain measured internal temperature — no sensor can measure it without destroying the cell. We compute `T_internal` using the **two-node thermal model** applied to the real measured current and surface temperature. We validate this by checking that our predicted surface temperature matches the real thermocouple measurement (mean RMSE 0.187°C, 100% of runs under 1.5°C).

---

## 6. Model Architecture

### LSTM / GRU Architecture (identical structure, different cell type)

```
Input: [batch_size, 60 timesteps, 12 features]
           │
    ┌──────▼──────┐
    │  LSTM/GRU   │  Layer 1: 256 hidden units
    │  Layer 1    │
    └──────┬──────┘
    ┌──────▼──────┐
    │  Dropout    │  20% during training
    └──────┬──────┘
    ┌──────▼──────┐
    │  LSTM/GRU   │  Layer 2: 256 hidden units
    │  Layer 2    │
    └──────┬──────┘
    ┌──────▼──────┐
    │  Dropout    │  20% during training
    └──────┬──────┘
    ┌──────▼──────┐
    │  Linear     │  256 → 64
    │  + ReLU     │
    └──────┬──────┘
    ┌──────▼──────┐
    │  Linear     │  64 → 1
    └──────┬──────┘
           │
    Output: predicted T_internal (°C)
```

**Total parameters:** 819,329 (LSTM) | ~615,000 (GRU)

### The 12 Input Features

| # | Feature | Description |
|---|---|---|
| 1 | `current_A` | Battery current in Amperes (positive = discharge) |
| 2 | `voltage_V` | Terminal voltage in Volts |
| 3 | `SOC` | State of Charge (0.0 = empty, 1.0 = full) |
| 4 | `T_ambient_C` | Surrounding air temperature in °C |
| 5 | `T_surface_C` | Surface thermocouple measurement in °C |
| 6 | `Q_gen_W` | Estimated heat generation rate in Watts |
| 7 | `dI_dt` | Rate of current change per second |
| 8 | `dV_dt` | Rate of voltage change per second |
| 9 | `delta_T_surf` | Surface temperature minus ambient temperature |
| 10 | `I_mean_60s` | Rolling 60-second average current |
| 11 | `I_std_60s` | Rolling 60-second current standard deviation |
| 12 | `R_internal_Ohm` | Estimated internal resistance in Ohms |

### PINN Loss Function

```
Total Loss = Huber(predicted, true) + λ × physics_residual

Where:
  Huber(predicted, true)  = data fitting loss
  physics_residual        = |mCp_core × dTcore/dt − Q_gen + k × (Tcore − Tsurf)|
  λ                       = 0.000001 (start) → 0.0001 (end, ramped over training)
```

The physics residual penalises any prediction that violates the two-node heat transfer equations, ensuring thermodynamically consistent outputs.

---

## 7. Repository Structure

```
battery-internal-temperature-prediction/
│
├── README.md                        ← You are here
├── BELIV_Lab_Battery_Guide.docx     ← Complete step-by-step lab guide
├── .gitignore                       ← Excludes large data files
│
├── calb_preprocess.py               ← STEP 1: Load .mat files, compute T_internal
├── battery_train_p2.py              ← STEP 2: Train Baseline LSTM + LSTM+PINN
├── battery_train_gru.py             ← STEP 3: Train GRU comparison model
├── battery_validate_p2.py           ← STEP 4: Validate all 3 models together
├── battery_lstm_matlab.m            ← MATLAB: Independent cross-framework validation
│
├── check_mat.py                     ← Utility: Inspect raw .mat file structure
├── check_data.py                    ← Utility: Check processed dataset statistics
└── analyze_calb.py                  ← Utility: Analyse dataset distributions

NOT in repository (generated by running the scripts):
├── calb_data/                       ← Generated by calb_preprocess.py
│   ├── calb_train.csv               (4.76M rows)
│   ├── calb_val.csv                 (1.03M rows)
│   ├── calb_test.csv                (1.04M rows)
│   ├── calb_full_dataset.csv        (6.83M rows)
│   └── surface_validation.png
├── results_p2/                      ← Generated by battery_train_p2.py
│   ├── p2_baseline_lstm.pt          (model weights)
│   ├── p2_pinn_lstm.pt              (model weights)
│   └── validation/
│       ├── time_series_all_models.png
│       ├── scatter_plot.png
│       ├── error_distribution.png
│       ├── rmse_by_profile.png
│       ├── rmse_by_temperature.png
│       ├── rmse_heatmap.png
│       └── validation_report.txt
└── results_gru/                     ← Generated by battery_train_gru.py
    ├── gru_model.pt
    └── gru_metrics.txt
```

---

## 8. Getting Started

### Requirements

```bash
Python 3.9+
CUDA 11.8+ (optional but recommended — 10x faster training)
```

### Install Dependencies

```bash
# With GPU (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Without GPU
pip install torch torchvision torchaudio

# All other dependencies
pip install numpy pandas scipy scikit-learn matplotlib
```

### Download the Dataset

1. Go to https://data.mendeley.com/datasets/ycx459r5c3/2
2. Download and extract the ZIP file
3. Place the three temperature folders inside your `battery_data` directory:

```
battery_data/
    Temperature_10C/
    Temperature_25C/
    Temperature_40C/
    calb_preprocess.py
    battery_train_p2.py
    ... (all scripts)
```

---

## 9. Step-by-Step Pipeline

Run these four scripts in order. Each step depends on the previous one completing successfully.

### Step 1 — Preprocess the Data (5 to 10 minutes)

```bash
cd C:\Users\YourName\battery_data
python calb_preprocess.py
```

**What it does:**
- Loads all 165 `.mat` files from the CALB dataset
- Resamples from ~0.1 second to 1 second intervals
- Computes SOC via Coulomb counting from real current measurements
- Applies the two-node thermal model to compute `T_internal`
- Engineers all 12 input features
- Splits 164 runs into train (114) / val (25) / test (25)

**Expected output:**
```
SURFACE TEMPERATURE VALIDATION
Mean RMSE    : 0.187 C
Under 1.5C   : 100.0% of runs
STATUS: VALIDATED -- thermal model accepted

DATASET SUMMARY
Total rows   : 6834132
Total runs   : 164
```

---

### Step 2 — Train Baseline LSTM and LSTM+PINN (60 to 90 minutes)

```bash
python battery_train_p2.py
```

**What it does:**
- Trains Baseline LSTM (no physics constraint)
- Trains LSTM+PINN (with two-node physics penalty in loss function)
- Saves both model weights to `results_p2/`

**Expected output:**
```
MODEL 1: Baseline LSTM
  RMSE = 0.094 C
  MAE  = 0.069 C

MODEL 2: LSTM + PINN
  RMSE = 0.116 C
  MAE  = 0.089 C
```

---

### Step 3 — Train GRU Comparison Model (45 to 60 minutes)

```bash
python battery_train_gru.py
```

**What it does:**
- Trains GRU with identical hyperparameters to LSTM
- Provides architectural comparison (fewer parameters, similar accuracy)

**Expected output:**
```
GRU Test Results:
  RMSE = 0.114 C
  MAE  = 0.091 C
```

---

### Step 4 — Validate All Three Models (5 to 10 minutes)

```bash
python battery_validate_p2.py
```

**What it does:**
- Loads all three saved models
- Evaluates on the held-out test set (25 runs, 1.04M rows)
- Generates 6 paper-ready figures
- Saves `validation_report.txt` with all numbers

**Expected output:**
```
Baseline LSTM : RMSE=0.0990  MAE=0.0740
GRU           : RMSE=0.1140  MAE=0.0905
LSTM+PINN     : RMSE=0.1160  MAE=0.0890
```

---

## 10. Understanding the Output

### validation_report.txt
Contains every number needed for the paper — overall metrics, breakdown by drive cycle profile, breakdown by ambient temperature, and comparison against published baselines.

### time_series_all_models.png
The key paper figure. Shows all three model predictions plotted against the true T_internal over time for one WLTP run. Closer to the black ground truth line = more accurate.

### scatter_plot.png
Predicted vs true temperature for all three models. A perfect model would have all dots on the diagonal line.

### rmse_by_profile.png
Bar chart showing RMSE for each drive cycle (WLTP, UDDS, US06, C20_charge, C20_discharge) for all three models side by side.

### rmse_by_temperature.png
Bar chart showing RMSE at 10°C, 25°C, and 40°C ambient. Shows how well each model generalises across thermal conditions.

### rmse_heatmap.png
Colour grid of RMSE for every combination of profile and temperature. Green = accurate, red = less accurate.

---

## 11. Physics Model

We use the **two-node lumped thermal model** (Forgez et al., 2010 — cited 500+ times) to compute T_internal from real measured data.

```
Core node:    mCp_core × dTcore/dt = Q_gen − k × (Tcore − Tsurf)
Surface node: mCp_surf × dTsurf/dt = k × (Tcore − Tsurf) − hA × (Tsurf − Tamb)

Heat generation: Q_gen = I² × R_internal
Resistance:      R = 0.0008 + 0.0005×(1−SOC) + 0.0003×exp(−10×(SOC−0.05))
```

### Thermal Parameters

| Parameter | Value | Description |
|---|---|---|
| `mCp_core` | 650 J/K | Core thermal mass |
| `mCp_surf` | 280 J/K | Surface thermal mass |
| `k` | 1.2 W/K | Core-to-surface thermal conductance |
| `h_conv` | 12.0 W/m²K | Convection coefficient |
| `A_surf` | 0.052 m² | Cell surface area |
| `hA` | 0.624 W/K | Convective heat loss coefficient |

### Validation Method
Since T_internal cannot be directly measured, we validate the physics model by checking that its predicted T_surface matches the **real thermocouple measurement** from the CALB dataset. Mean RMSE of 0.187°C across all 164 runs confirms the model correctly captures the thermal behaviour of this specific cell.

---

## 12. Comparison Against Published Work

| Study | Cell Type | Chemistry | T_internal Source | RMSE (°C) |
|---|---|---|---|---|
| Wang et al. 2021 | Cylindrical 18650 | LFP | Physical sensor | 0.850 |
| Karnehm et al. 2024 LSTM | Cylindrical 21700 | NMC | Physical sensor | 0.854 |
| Karnehm et al. 2024 KAN | Cylindrical 21700 | NMC | Physical sensor | 0.751 |
| Shen et al. 2025 PINN | Blade 150Ah | LFP | Three-node model | 0.570 |
| **Ours — Baseline LSTM** | **Prismatic 58Ah** | **NMC** | **Two-node model** | **0.099** |
| Ours — GRU | Prismatic 58Ah | NMC | Two-node model | 0.114 |
| Ours — LSTM+PINN | Prismatic 58Ah | NMC | Two-node model | 0.116 |

### Why Our RMSE Is Lower — Important Context
Our superior RMSE compared to published work is partly explained by:
1. **Narrower temperature range** — our data covers 6.84°C to 42.36°C (35°C range) vs 60°C+ in fast-charge studies
2. **Larger dataset** — 6.83M rows vs typically tens of thousands in published work
3. **Derived ground truth** — T_internal comes from the same physics model used as the PINN constraint, creating mathematical consistency

These factors are acknowledged as limitations. The genuine novel contribution is the systematic validation of LSTM, GRU, and LSTM+PINN on real experimental drive cycle data for large-format prismatic NMC cells, which has not been previously published.

---

## 13. Known Limitations

- **No extreme fast charging** — the CALB dataset does not include 2C/3C fast charge scenarios. Temperatures never exceed 42.36°C. The model's performance at temperatures above 45°C (the thermal runaway risk zone) has not been validated.
- **Derived T_internal** — ground truth internal temperature is computed from the two-node model rather than physically measured with an invasive sensor. While validated via surface temperature comparison, this introduces a degree of circularity.
- **Single cell chemistry** — validated only on NMC prismatic cells. Performance on LFP, LCO, or other chemistries is unknown.
- **Controlled lab conditions** — real-world BMS deployment may encounter sensor noise, calibration drift, and current profiles outside the training distribution.

---

## 14. For New Lab Members

The complete step-by-step guide including plain-English explanations of every concept, troubleshooting table, and terminology glossary is in:

**`BELIV_Lab_Battery_Guide.docx`**

Quick start checklist:
- [ ] Install Python 3.9+ and PyTorch
- [ ] Download CALB dataset from Mendeley
- [ ] Run `python calb_preprocess.py`
- [ ] Run `python battery_train_p2.py`
- [ ] Run `python battery_train_gru.py`
- [ ] Run `python battery_validate_p2.py`
- [ ] Check `results_p2/validation/validation_report.txt` for all results

**Training time estimate:** ~3 hours total on RTX 4060 GPU | ~12 hours on CPU

---

## References

1. Forgez, C., Do, D.V., Friedrich, G., Morcrette, M., Delacourt, C. (2010). Thermal modeling of a cylindrical LiFePO4/graphite lithium-ion battery. *Journal of Power Sources*, 195(9), 2961–2968.
2. Raissi, M., Perdikaris, P., Karniadakis, G.E. (2019). Physics-informed neural networks. *Journal of Computational Physics*, 378, 686–707.
3. Karnehm, D., et al. (2024). Internal temperature prediction using KAN and LSTM. *IEEE Transactions*, 2024.
4. Shen, L., et al. (2025). Physics-informed LSTM for blade battery thermal estimation. *Applied Thermal Engineering*, 2025.
5. CALB L148N58A Dataset — University of Trieste. Mendeley Data: https://data.mendeley.com/datasets/ycx459r5c3/2

---

## Lab

**BELIV Lab**
Graduate Research Assistant — April 2025 to Present

*Research focuses on physics-informed machine learning for battery thermal safety in electric vehicle applications.*

---

*Last updated: April 2026*
