# NLP Salary Prediction Project

## Project Overview
This project is an end-to-end Machine Learning pipeline that predicts job salaries based on natural language job descriptions and categorical features (such as location, company, and contract type). It utilizes a custom PyTorch neural network architecture that combines text embeddings with categorical embeddings.

## Codebase Structure
The project follows a modular architecture, separating the configuration, data processing, modeling, and execution layers.

```text
NLP_PROJECT/
├── scripts/                     # Executable scripts for the training and prediction pipeline
│   ├── main.py                  # Entry point for training the model
│   ├── predict.py               # Entry point for data for prediction
├── artifacts/                   # Saved model weights (.pth) and preprocessor state (.pkl)
├── data/                        # Raw training data and inference inputs/outputs
├── src/                         # Core logic modules
│   ├── models/
│   │   └── self_taught_net.py   # PyTorch Dataset and Neural Network architecture
│   ├── pipeline/
│   │   ├── operations.py        # Training loop, evaluation, and MLflow logging
│   │   └── preprocessing.py     # Text/categorical transformation and vocabulary building
│   └── utils.py                 # Helper functions (custom logging, YAML config loader)
├── .gitignore                   # Git exclusion rules
├── config 
│   ├── config.yaml              # Centralized configuration (paths, architecture)
├── documentation/               
│   ├── documentation.txt        # Project analysis and observations 
└── requirements.txt             # Project dependencies
├── notebook/                    
│   ├── notebook_experiments.ip  # Jupyter notebook for model experimentation and analysis
└── requirements.txt             # Environment requirements
```

## How to Run the Project

### 1. Installation
Install the required dependencies using:

```bash
pip install -r requirements.txt
```

### 2. Download data
Download the [Job Salary Prediction](https://www.kaggle.com/c/job-salary-prediction/data) dataset from Kaggle. 
Place the `Train_rev1.csv` file inside the `data/` folder located in the root directory of the project.


### 3. Configuration
Open config.yaml to adjust your settings before starting the training.

### 4. Training
Run the training script. The system will automatically log metrics and save the best model. Once finished, copy the run_id displayed in the console.

```bash
python3 -m scripts.main
```

### 5. Setup for Prediction
Go back to config.yaml and paste the copied ID into the run_id field to tell the system which model version to use:

```YAML
mlflow:
  run_id: "YOUR_COPIED_ID_HERE"
```

### 6. Prediction
Run the prediction script to generate results based on the selected model:

```bash
python3 -m scripts.predict
```


