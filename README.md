# NLP Salary Prediction Project

## Project Overview
This project is an end-to-end Machine Learning pipeline that predicts job salaries based on natural language job descriptions and categorical features (such as location, company, and contract type). It utilizes a custom PyTorch neural network architecture that combines text embeddings with categorical embeddings.

## Codebase Structure
The project follows a modular architecture, separating the configuration, data processing, modeling, and execution layers.

```text
NLP_PROJECT/
├── artifacts/               # Saved model weights (.pth) and preprocessor state (.pkl)
├── data/                    # Raw training data and inference inputs/outputs
├── src/                     # Core logic modules
│   ├── models/
│   │   └── self_taught_net.py # PyTorch Dataset and Neural Network architecture
│   ├── pipeline/
│   │   ├── operations.py      # Training loop, evaluation, and MLflow logging
│   │   └── preprocessing.py   # Text/categorical transformation and vocabulary building
│   └── utils.py             # Helper functions (custom logging, YAML config loader)
├── .gitignore               # Git exclusion rules
├── config.yaml              # Centralized configuration (hyperparameters, paths, architecture)
├── main.py                  # Entry point for training the model
├── predict.py               # Entry point for batch inference on new data
└── requirements.txt         # Project dependencies
