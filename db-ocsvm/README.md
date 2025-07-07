# DB-OCSVM
# Project Structure
    project_root/
    ├── data/
    │   ├── raw/                  # Original data
    │   ├── processed/            # Cleaned and preprocessed data
    │   ├── sample/               # sample size of the processed data, used for testing
    │
    ├── models/
    │   ├── base_models/         # Individual models before hybridization
    │   ├── hybrid_models/       # Hybrid model implementations
    │   └── saved_models/        # Trained model checkpoints
    │
    ├── src/
    │   ├── data/
    │   │   ├── __init__.py
    │   │   ├── data_loader.py   # Data loading utilities
    │   │   └── preprocessor.py  # Data preprocessing scripts
    │   │
    │   ├── features/
    │   │   ├── __init__.py
    │   │   └── feature_engineering.py
    │   │
    │   ├── models/
    │   │   ├── __init__.py
    │   │   ├── base_model.py    # Base model implementations
    │   │   ├── hybrid_model.py  # Hybrid model architecture
    │   │   └── model_utils.py   # Helper functions for models
    │   │
    │   ├── training/
    │   │   ├── __init__.py
    │   │   ├── trainer.py       # Training loop implementation
    │   │   └── validation.py    # Validation utilities
    │   │
    │   └── visualization/
    │       ├── __init__.py
    │       └── visualize.py     # Plotting and visualization tools
    │
    ├── notebooks/
    │   ├── exploratory/         # Jupyter notebooks for EDA
    │   ├── modeling/            # Model development notebooks
    │   └── evaluation/          # Performance analysis notebooks
    │
    ├── configs/
    │   ├── model_config.yaml    # Model hyperparameters
    │   ├── train_config.yaml    # Training configuration
    │   └── data_config.yaml     # Data processing parameters
    │
    ├── tests/
    │   ├── test_data.py
    │   ├── test_models.py
    │   └── test_training.py
    │
    ├── logs/                    # Training logs and TensorBoard files
    │
    ├── docs/                    # Documentation
    │   ├── api/
    │   ├── experiments/
    │   └── setup.md
    │
    ├── requirements.txt         # Project dependencies
    ├── setup.py                # Package installation script
    ├── README.md               # Project overview and instructions
    └── .gitignore             # Files to ignore in version control