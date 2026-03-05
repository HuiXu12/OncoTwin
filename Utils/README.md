# Digital Twins Survival Analysis — Demo Code

End-to-end pipeline for constructing and evaluating a Digital Twin (DT) 
survival model for Progression-Free Survival (PFS) prediction, including 
risk stratification, calibration, individual survival curve generation, 
and treatment effect estimation.

---

## Repository Structure

```
digital_twins_survival/
├── demo1_model_construction.py          # Part 1 – Feature engineering & XGBoost-AFT training
├── demo2_calibration_digital_twin.py    # Part 2 – Post-hoc calibration & DT curve generation
├── demo3_risk_stratification_visualization.py  # Part 3 – Risk stratification & figures
├── utils_survival.py                    # Shared KM / SHAP / stat utility functions
├── utils_calibration.py                 # Calibration pipeline (full_calibration_pipeline, etc.)
│                                        #   ← supply your own implementation
└── README.md
```

---

## Quick Start

### 1 – Install dependencies

```bash
pip install numpy pandas scikit-learn xgboost xgbse lifelines shap \
            matplotlib seaborn scipy dill openpyxl
```

GPU training requires an NVIDIA GPU with CUDA drivers installed.  
Set `"device": "cpu"` in `PARAMS_XGB_AFT` inside `demo1_model_construction.py` to run on CPU.

### 2 – Prepare your data

Each cohort requires **three Excel files** (pre-treatment, post-treatment, delta):

| File | Description |
|------|-------------|
| `VolOriginal_Base_before_<cohort>.xlsx` | Pre-treatment radiomic features |
| `VolOriginal_Base_after_<cohort>.xlsx`  | Post-treatment radiomic features |
| `VolOriginal_Base_delta_<cohort>.xlsx`  | Delta (change) radiomic features |

Each file must contain:
- A unique patient identifier column (e.g. `PatientID`)  
- Radiomic columns spanning from `Met_Bone` to `ATS_AllTumor`  
- Clinical columns: `Sex`, `Race`, `Ethnicity`, `Smoker`, `Pathology`, `Drug`, `RECIST_before`, `Age`, lab values  
- Outcome columns: `PFS` (time in months), `PFS_events` (0 = censored, 1 = event)

Update the `train_data_dir` and `test_data_dir` paths at the top of **Demo 1**.

### 3 – Run demos in order

```bash
python demo1_model_construction.py        # trains model, saves xgb_model.json + SHAP
python demo2_calibration_digital_twin.py  # calibrates, builds DT curves, estimates ITE
python demo3_risk_stratification_visualization.py  # risk groups, KM plots, SHAP figures
```

> **Tip:** Run interactively in JupyterLab by converting each script to a notebook  
> (`jupytext --to notebook demo1_model_construction.py`).

---

## Demo Descriptions

### Demo 1 — Model Construction (`demo1_model_construction.py`)

| Step | Description |
|------|-------------|
| Data loading | Read and merge pre/post/delta radiomic tables for train + test cohort |
| Categorical encoding | Binary-encode sex, race, ethnicity, smoking, pathology, RECIST |
| Feature filtering | Remove highly correlated radiomic features (Pearson \|r\| > 0.98) |
| XGBoost-AFT training | DART booster, GPU accelerated, 200 boosting rounds |
| Evaluation | Concordance index (C-index) on train and external test cohort |
| Feature importance | Gain-based bar chart + SHAP beeswarm summary |

**Key outputs:**  
`results/model_output/xgb_model.json`  
`results/model_output/feature_importance_gain.*`  
`results/model_output/SHAP/SHAP_summary_*.png`

---

### Demo 2 — Calibration & Digital Twin (`demo2_calibration_digital_twin.py`)

| Step | Description |
|------|-------------|
| Drug-stratified calibration | Fit Weibull / log-normal calibrators per drug arm using `full_calibration_pipeline` |
| Brier score | Compute and plot time-varying Brier score + IBS |
| Individual DT curves | Plot per-patient predicted survival curves with median / mean annotations |
| ITE estimation | Estimate ΔRMST (LCT vs. no-LCT) per test patient via `estimate_LCT_ITE_for_test` |
| Counterfactual plot | Plot a selected patient's LCT vs. no-LCT counterfactual survival curves |
| Export | Save Digital Twin PFS predictions and ITE table to CSV |

**Key outputs:**  
`results/model_output/Calibration_*/`  
`results/model_output/ITE_test_cohort_60mo.csv`  
`results/model_output/digital_twin_predictions.csv`

---

### Demo 3 — Risk Stratification & Visualization (`demo3_risk_stratification_visualization.py`)

| Step | Description |
|------|-------------|
| Optimal cutoff | Log-rank search over percentile grid on training predictions |
| Risk groups | Assign High / Low risk labels to train and test cohorts |
| KM plots | Plot KM curves for risk groups; Digital Twin vs. Observed; four-curve overlays |
| HR computation | Cox proportional hazards HR (High vs. Low; DT vs. Observed) |
| SHAP aggregation | Overall beeswarm + correlation-filtered top-15 plot |
| Top-6 bar chart | Publication-style horizontal bar chart of top SHAP features |
| ITE correlation | Spearman ρ between features and ΔRMST, annotated with significance stars |

**Key outputs:**  
`results/model_output/KM_*.png`  
`results/model_output/SHAP_summary_overall.png`  
`results/model_output/SHAP_top6_features.png`  
`results/model_output/ITE_feature_correlation.png`  
`results/model_output/cox_*.csv`

---

## Utility Modules

### `utils_survival.py`

Contains all KM plotting, statistical comparison, and SHAP utility functions
used across the three demos.  No patient-specific data or file paths are
hard-coded in this module.

Key functions:

| Function | Description |
|----------|-------------|
| `km_analysis` | Compute median PFS and N-month survival with 95% CI for up to four groups |
| `remove_highly_correlated_features` | Pearson correlation pruning |
| `plot_km_curve` | Two-arm KM plot with at-risk table |
| `plot_km_curve_dashed` | Two-arm KM plot with configurable line styles |
| `plot_multi_km` | Multi-arm KM plot with inline at-risk counts |
| `compare_low_high` | Cox HR + log-rank p for two survival groups |
| `safe_shap_summary_plot` | Memory-safe SHAP beeswarm + importance CSV |
| `knn_predict_pfs` | KNN-based PFS prediction from risk scores |
| `best_cutoff_per_drug` | Per-drug optimal log-rank cutoff |
| `compare_cindex_from_ci` | Approximate ΔC-index significance test from CIs |

### `utils_calibration.py`

Calibration-specific utilities (not included in this repository).  
The following functions are imported in Demo 2:

- `full_calibration_pipeline` — fits drug-stratified survival calibrators  
- `plot_ibs_curve_DT` — plots time-varying Brier score  
- `plot_patients_survival_DT_split_legend` — plots individual DT survival curves  
- `estimate_LCT_ITE_for_test` — estimates per-patient ITE (ΔRMST) for LCT  
- `plot_lct_counterfactual_for_patient` — counterfactual plot for one patient  

---

## Citation

If you use this code, please cite the corresponding manuscript (details to be
added upon publication).

---

## License

This code is released for research and educational use.  
See `LICENSE` for details.
