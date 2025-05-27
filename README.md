# Active Imputation

## Overview
This repository contains the code for the paper "Confidence-Based Active Data Imputation via Distribution
Matching". 

## Introduction
Handling missing data is critical for building reliable machine learning models. While automated imputation methods are common, they often overlook the value of human expertise—especially in cases of high uncertainty.

This project introduces an **active imputation framework** that uses confidence-based guidance to involve human input effectively. We estimate confidence by measuring the variability of Optimal Transport (OT) imputations and propose an efficient approximation using **dropout noise injection**.

We implement two active strategies:
- **Batch-Impute**: Queries uncertain values in a single step.
- **Iter-Impute**: Distributes queries across multiple iterations for progressive refinement.

Experiments across five real-world datasets show that **Iter-Impute outperforms both baseline and random querying methods**, achieving better accuracy with limited user interaction. This makes our approach well-suited for practical, semi-automated data cleaning.



## Data Preparation

All datasets are automatically downloaded using `fetch_openml` from `scikit-learn`. No manual download is needed.

We used the following real-world datasets in our experiments:
- **Adult**
- **Wine**
- **German Credit**
- **Diabetic**
- **Breast Cancer**

### Preprocessing Steps:
1. **Feature Scaling**:  
   All numerical features are scaled to the \[0, 1\] range using `MinMaxScaler` from `scikit-learn`.

2. **Missing Value Injection**:  
   Missing values are injected artificially using the `Inject_Missing_Value.py` script.  
   This includes support for:
   - MCAR (Missing Completely at Random)
   - MAR (Missing At Random)
   - MNAR (Missing Not At Random)


## Experiments

All experiments are conducted in the `notebooks/` folder using Jupyter notebooks, with one notebook dedicated to each dataset.

### Brier Score Evaluation
- The notebook `experiment.ipynb` calculates the **Brier score** across all datasets to evaluate the confidence calibration of the imputation algorithms.

### Baseline Imputation Methods

Each dataset-specific notebook begins by applying the following three core methods under various missingness mechanisms (MCAR, MAR, MNAR):**OT-Impute**, **OT-Rand**, **Batch-Impute**
These three methods are first compared to evaluate the effectiveness of active selection over random or passive strategies.

In a later section of the notebook, these methods are further compared with **additional baseline imputation techniques**, including:
- Mean/Median imputation
- MICE
- KNN-Impute

### Optimization: Dropout Noise Sensitivity
- To improve confidence estimation in Batch-Impute, we inject different levels of **dropout noise** during the OT optimization process.
- We empirically evaluate how the choice of dropout noise affects imputation quality.
- Based on these experiments, the best-performing dropout level is selected.

### Iterative Active Imputation (Iter-Impute)
- In the later sections of each notebook, **Iter-Impute** is executed across multiple iterations.
- This method simulates active user involvement over time, gradually refining the imputation by querying the most uncertain values at each step.

### Visualization
- At the end of each notebook, all results are visualized.
- Plots include:
  - MAE comparisons across methods
  - Runtime analysis
  - Dropout noise impact
  - Performance across different iteration steps and budgets

> All figures and result CSVs are saved in the `Output/` directory for easy access.


## Results
All csv files and figues are saved in the Output directory.

