# End-to-End Text Classification Pipeline

This repository contains a complete machine learning workflow for multiclass text classification, including data cleaning, feature engineering (TF–IDF), model training, cross-validation, and evaluation.

The project was originally developed as a CSC311 final group project and has been repackaged for public release with an emphasis on clarity, reproducibility, and clean structure.

## Report

📄 **Technical report (PDF):**  
[Project_Report.pdf](report/Project_Report.pdf)

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

## Running

### Training (dataset required locally)
Place the dataset in a local `data/` directory following the expected layout, then run:

```bash
python -m src.train.train
```

### Inference
If the saved artifacts in `models/` are present:

```bash
python pred.py
```

## Notes

- Dataset files are not included in this repository.
- This public release is intended to showcase an end-to-end ML workflow, not to distribute data.
