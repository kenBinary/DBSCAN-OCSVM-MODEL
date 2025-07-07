# DBSCAN-OCSVM Project

## OVERVIEW

With the rapid evolution of cyber threats, anomaly-based intrusion detection systems are essential for safeguarding network infrastructures. Traditional One-Class SVM (OCSVM) methods work well for detecting common patterns but struggle with rare attacks and require significant computational resources. This project introduces a hybrid model, DB-OCSVM, which combines DBSCAN clustering with One-Class SVM. The approach uses an Autoencoder for feature extraction and dimensionality reduction, DBSCAN for clustering, and SHAP-based Explainable AI for transparent decision-making. Experiments on the NSL-KDD and CIDDS-001 datasets show improved accuracy, F1-score, and significant reductions in training and inference times. The hybrid model is more effective at detecting unusual attacks, reduces false alarms, and provides clearer results, making intrusion detection systems more adaptable and effective.

## Repository Structure

- **db-ocsvm/**  
  Model development, training, and evaluation code.

  - Contains data loaders, preprocessing scripts, model definitions, training utilities, and Jupyter notebooks for exploratory analysis and model development.
  - Datasets (raw and processed) and model executables for different datasets (NSL-KDD, CIDDS-001) are included.
  - See `db-ocsvm/README.md` for a detailed breakdown.

- **db-ocsvm-simulator/**  
  Desktop simulator for running and evaluating the DB-OCSVM and OCSVM models.
  - Built with Electron and React for a user-friendly interface.
  - Allows you to run, test, and visualize model performance.
  - See `db-ocsvm-simulator/README.md` for setup and usage instructions.

## Installation

1. Clone the repository.
2. For model development, install Python dependencies:
   ```
   pip install -r db-ocsvm/requirements.txt
   ```
3. For the simulator, install Node.js dependencies:
   ```
   cd db-ocsvm-simulator
   npm install
   ```

## Usage

- For model training and evaluation, use scripts and notebooks in `db-ocsvm/`.
- To launch the simulator desktop app:
  ```
  cd db-ocsvm-simulator
  npm run dev
  ```
- For more options, see the individual `README.md` files in each subfolder.

## Requirements

- Python >= 3.8 (for model development)
- Node.js & npm (for simulator)
- See each subfolder's `requirements.txt` or `package.json` for details.

## Authors

- JAMES PAUL S. BALAGAT
- KENNETH JOSHUA D. BECARO
- JEPHUNNEH DENIEL D. SANTIAGO

## Acknowledgements

We extend our deepest gratitude to God for granting us the strength and perseverance to complete this thesis.

We sincerely thank our adviser, Sir Ramcis, for his unwavering support, guidance, and valuable insights, which were instrumental in the successful completion and defense of our study.

We also thank our panel members—Sir MJ, Ma’am Yara, and Sir Hernando—for their constructive feedback and recommendations that greatly enhanced the quality of our research.

We are grateful to our beta testers—PMAJ Julius P. Santillana, Mr. Jourdan Salazar, and Mr. Jan Carlo T. Arroyo, DIT—for their technical assistance and valuable input.

We appreciate our classmates who participated and supported our research despite their own academic responsibilities.

We also acknowledge the authors of related studies whose work served as both foundation and inspiration for our thesis.

Lastly, we thank one another for our commitment and teamwork, and our parents for their unwavering love and support throughout this journey.

This work is dedicated to all who made its completion possible.
