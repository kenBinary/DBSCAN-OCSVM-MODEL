# DB-OCSVM

## Overview

DB-OCSVM is a hybrid anomaly detection framework that combines Density-Based Spatial Clustering of Applications with Noise (DBSCAN) and One-Class Support Vector Machine (OCSVM) for network intrusion detection. The project is designed to preprocess, train, and evaluate models on datasets such as CIDDS-001 and NSL-KDD, providing both classical and deep learning-based approaches.

## Folder Structure

```
constants/                # Dataset paths and constants
    __init__.py
    dataset_paths.py

data/                     # Datasets for experiments
    raw/                  # Original, unprocessed data
        CIDDS-001/
        NSL-KDD/
    processed/            # Preprocessed data for training/testing
        CIDDS-001/
        NSL-KDD/

model-executables/        # Model scripts and artifacts for inference
    cidds-001/
        base/
        proposed/
    nsl-kdd/
        base/
        proposed/

notebooks-preprocessing/  # Jupyter notebooks for data preprocessing and exploration
    data/
        CIDDS-001/
        NSL-KDD/
        util/
    exploratory/
        CIDDS-001/
        NSL-KDD Dataset/
    training/
        CIDDS-001/
        NSL-KDD/

notebooks-training/       # Jupyter notebooks for model training and inference
    cidds-001/
        autoencoder/
        notebooks/
        notebooks-inference/
    nsl-kdd/
        autoencoder/
        notebooks/
        notebooks-inference/

LICENSE                   # License file
README.md                 # Project documentation (this file)
requirements.txt          # Python dependencies
```

## Setup & Installation

1. **Clone the repository:**
   ```sh
   git clone <repo-url>
   cd db-ocsvm
   ```
2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## Usage

### Data Preprocessing

- Use the notebooks in `notebooks-preprocessing/data/` to preprocess raw datasets. Utilities for data manipulation are in `util/`.
- Preprocessed data is saved in `data/processed/`.

### Model Training

- Training notebooks are in `notebooks-training/<dataset>/`.
- You can train autoencoders, OCSVM, and hybrid models using these notebooks.
- Model artifacts are saved in `model-executables/<dataset>/<model-type>/`.

### Inference

- Use the `inference-pipeline.py` scripts in `model-executables/<dataset>/<model-type>/` to run inference on test data.
- Example:
  ```sh
  python model-executables/cidds-001/proposed/inference-pipeline.py
  ```

## Main Components

- **constants/**: Centralized dataset paths and configuration constants.
- **data/**: Contains both raw and processed datasets for experiments.
- **model-executables/**: Scripts and trained models for running inference.
- **notebooks-preprocessing/**: Jupyter notebooks for data cleaning, feature engineering, and exploratory analysis.
- **notebooks-training/**: Jupyter notebooks for model training and evaluation.

## Datasets

- **CIDDS-001**: Located in `data/raw/CIDDS-001/` and `data/processed/CIDDS-001/`.
- **NSL-KDD**: Located in `data/raw/NSL-KDD/` and `data/processed/NSL-KDD/`.

## Requirements

All dependencies are listed in `requirements.txt`. Install with pip as shown above.
