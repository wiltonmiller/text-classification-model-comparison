# Text Classification Pipeline

This repo contains a complete machine learning workflow for a multiclass text classification problem, including data cleaning, feature engineering (TF–IDF), model training, cross-validation, and evaluation.

The project was initially developed as a CSC311 final group project and has been repackaged for public release with an emphasis on reproducibility and a clean structure.

## Report

📄 **Technical report (PDF):**  
[Project_Report.pdf](report/Project_Report_CSC311.pdf)

## Repository Structure

- `src/` — core pipeline code (cleaning, features, training utilities)
- `src/train/train.py` — training entry point
- `pred.py` — inference / prediction entry point
- `models/` — saved preprocessing + model artifacts
- `final_pred/` — final prediction scripts
- `report/` — project report PDF

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

This repository does not include the dataset files.

To reproduce the pipeline, place the raw dataset at the following path:

```
data/raw/training_data.csv
```

The preprocessing pipeline will generate the following directories and files:

```
data/cleaned/
  ├── training_data_clean.csv
  └── testing_data_clean.csv

data/processed/
  ├── training_features.npy
  ├── testing_features.npy
  ├── training_labels.npy
  └── testing_labels.npy
```

Saved preprocessing artifacts (e.g., vectorizers and metadata) are written to:

```
models/
```

Dataset files are not included due to redistribution constraints.

## Running

### Preprocessing
Clean and split the raw dataset:

```bash
python -m src.data.cleaning
```

Extract features and save preprocessing artifacts:

```bash
python -m src.data.features
```

### Training
Train the model using the processed features:

```bash
python -m src.train.train
```

### Inference
If the saved artifacts in `models/` are present, run:

```bash
python pred.py
```

## Notes

- Dataset files are not included in this repository.
- This public release is intended to showcase an end-to-end ML workflow and reproducible pipeline structure, not to distribute data.
