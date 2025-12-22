# OncoTwin
OncoTwin is a generalizable multimodal digital-twin framework to model individualized treatment trajectories by integrating imaging-derived tumor burden dynamics with routine blood test and demographic variables in ALK-rearranged NSCLC. Across systematic validation spanning real-world datasets and prospective clinical trials, OncoTwin accurately reproduced survival outcomes across TKI generations, simulated virtual control arm, and estimated patient-level treatment effects. Its successful deployment in the Phase II BRIGHTSTAR trial demonstrates the feasibility and translational potential of AI-enabled digital twins and establishes a blueprint for broader applications in oncology. 

![Pipeline overview](figures/pipeline.png)

## Key Features
1.	Digital twin bridging real-world and trial data: OncoTwin integrates real-world data from MD Anderson with two prospective trial cohorts (Phase III ALTA-1L and Phase II BrightStar) to enable robust risk stratification, calibrated individualized survival prediction, and virtual control-arm simulation in clinical trials.
2.	Multimodal, longitudinal modeling with clinically accessible inputs: OncoTwin lev-erages whole-body 3D tumor burden, routine blood tests, and clinical demographics across baseline, early on-treatment, and delta timepoints, yielding interpretable and readi-ly translatable clinical insights.
3.	Trial-integrated clinical applications: Within the Phase II BrightStar trial, OncoTwin simulated a brigatinib-only control arm to estimate the treatment effect of LCT, and used simulation-based resampling to determine stable sample size thresholds. Individualized treatment-effect estimates further highlight its potential for adaptive and evidence-guided trial design.
4.	Biological and therapeutic insight: Beyond aggregate endpoints, OncoTwin revealed early-response heterogeneity between TKI generations, with second-generation TKIs achieving superior whole-body and organ-specific responses, and enabled TKI-specific risk stratification to support more granular treatment strategies.
5.	Spatially resolved disease modeling:  By capturing inter-patient differences, OncoT-win quantified not only whole-body but also organ-specific tumor burden dynamics, providing a foundation for organ adaptive therapeutic strategies and a deeper under-standing of site-specific resistance.

## Installation
To install the development version of OncoTwin using pip, run the following command:
```bash
pip install git+https://github.com/WuLabMDA/OncoTwin.git
```

## The repository contains the following files:
### 1. Model Training
- File: xxx.py
- Description: Trains the machine learning models using repeated cross-validation, bootstrapping, feature selection, and survival analysis. Implements Grey Wolf Optimizer for feature reduction.
- Inputs: ISABR_trial.csv, matched_id.csv, unmatched_id.csv
- Outputs: Trained models and cross-validation results.









