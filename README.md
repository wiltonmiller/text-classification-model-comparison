# CSC311 Group Project

This repository contains our final project for CSC311. 
It includes the full codebase for our machine learning pipeline, including data preprocessing, analysis, model development, and experiments.

## Project Structure

    .
    ├── data/                # raw and processed data (ignored in .gitignore)
    │   ├── raw/
    │   └── processed/
    ├── src/                # main source code
    │   ├── data/           # data loading and cleaning
    │   ├── train/          # training scripts
    │   ├── analysis/       # exploratory analysis
    │   ├── models/         # model implementations
    │   ├── utils/          # helper functions
    │   └── examples/       # starter files and prototypes
    ├── models/             # saved model artifacts
    ├── final_pred/         # final prediction script and outputs
    ├── report/             # project report files
    ├── WORKFLOW.md         # internal collaboration workflow
    └── README.md

## Getting Started

Clone the repository:

    git clone git@github.com:wiltonmiller/csc311-2025-group35900.git
    cd csc311-2025-group35900

Install dependencies (requirements.txt will be added when finalized):

    pip install -r requirements.txt

## Team

- Wilton Miller  
- Benjamin Gavriely  
- Christopher Marrella  

## Overview

The project focuses on building a complete ML workflow:
- preprocessing and cleaning the dataset  
- exploratory data analysis  
- training and evaluating several models  
- selecting the best-performing model  
- generating final predictions  

## Notes

Data files are ignored by .gitignore and should be stored locally. 
This repository contains all development work and code used throughout the project.
