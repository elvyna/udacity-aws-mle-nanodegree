# Predicting E-Commerce Cart Abandonment

## Overview

This project is conducted as the capstone project of Udacity AWS Machine Learning Engineer Nanodegree. The dataset is taken from [Data Mining Cup 2013](https://www.data-mining-cup.com/reviews/dmc-2013/), which contains 429,013 rows of e-commerce sessions with 24 columns.

The repository is structured as follows.

```
├── README.md          <- description of the repo
|
├── requirements.in    <- list of packages used in this project
|
├── config             <- config files used on the python scripts
|
├── img                <- screenshots for the README
|
├── notebook           <- supporting notebooks for exploration etc.
|
├── proposal           <- files related to the project proposal
|
└── src                <- scripts
```

## Project Setup

### Local

For local development, you can use `conda` or `venv`.
```sh
conda install --force-reinstall -y -q --name <env-name> -c conda-forge imbalanced-learn --file requirements.in
```

```sh
python3 -m venv work-env
source work-env/bin/activate
pip install pip-tools
pip-compile
pip-sync
```

### AWS

**TO DO**
- S3 bucket preview - data store
- `notebook/sagemaker` for hp tuning, training, and deploy
- TO DO: prepare step functions to trigger endpoint?

## Future Improvement

### Modelling

- ...
- ...

### ML Ops

- Orchestrate the end-to-end scripts as step functions (require containerization) or as sagemaker pipeline (run them as sagemaker processing job)
- Model monitoring and retraining