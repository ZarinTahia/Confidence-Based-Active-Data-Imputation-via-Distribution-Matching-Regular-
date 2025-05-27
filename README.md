# Active Imputation

## Overview
This repository contains the code for the paper "Confidence-Based Active Data Imputation via Distribution
Matching". 

## Introduction
## Requirments
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
torch>=1.10.0
geomloss>=0.2.5
tqdm>=4.62.0
openml>=0.12.2


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
Brier score:
OT-Impute, OT-Rand, Batch-Impute:
Optimization:
Iter-Impute:
Visualization: 

## Results
All csv files and figues are saved in the Output directory.

