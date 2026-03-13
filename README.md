# Artificial Neural Network — Smoking Status Classification

A binary classification project that predicts an individual's smoking status from health and biometric data using a custom feedforward Artificial Neural Network (ANN) implemented from scratch.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Experimental Scenarios](#experimental-scenarios)
- [Classification Results](#classification-results)
- [Project Structure](#project-structure)
- [License](#license)

---

## Overview

This project addresses the task of classifying individuals as **smokers** or **non-smokers** based on physical measurements and biochemical markers. Smoking is a major risk factor for numerous health conditions; accurate prediction from routine health data can support screening and public health applications.

### Classification Task

| Class | Label | Description |
|-------|-------|-------------|
| 0 | Non-smoker | Individual does not smoke |
| 1 | Smoker | Individual smokes |

### Methodology

- **Forward Propagation**: Computation from input to output layer
- **Backward Propagation**: Weight updates via error backpropagation
- **Activation Function**: Sigmoid across all layers
- **Loss Function**: Mean Squared Error (MSE)

---

## Dataset

- **Source**: [Kaggle — Smoking and Drinking Dataset with Body Signal](https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset)
- **Type**: Health and biometric data
- **Samples**: 55,692
- **Features**: 26 (after dropping ID)

### Feature Categories

| Category | Examples |
|----------|----------|
| Demographics | gender, age |
| Anthropometrics | height, weight, waist |
| Sensory | eyesight (left/right), hearing (left/right) |
| Cardiovascular | systolic, relaxation (diastolic) |
| Metabolic | fasting blood sugar, cholesterol, triglycerides, HDL, LDL |
| Biochemical | hemoglobin, urine protein, serum creatinine, AST, ALT, Gtp |
| Other | oral, dental caries, tartar |

### Target Variable

- **smoking**: Binary (0 = Non-smoker, 1 = Smoker)

### Preprocessing

- **Categorical encoding**: `gender`, `oral`, `tartar` encoded via `LabelEncoder`
- **Standardization**: All features scaled with `StandardScaler` (zero mean, unit variance)

---

## Architecture

The ANN is a fully connected feedforward network with the following topology:

```
Input (25) → Hidden 1 (128) → Hidden 2 (64) → Output (1)
```

### Hyperparameters (Grid Search Optimal)

| Parameter | Value |
|-----------|-------|
| Hidden layers | (128, 64) |
| Learning rate | 0.5 |
| Iterations (epochs) | 150 |

### Grid Search Space

- **Hidden layers**: (32, 16), (64, 32), (128, 64)
- **Learning rates**: 0.1, 0.3, 0.5
- **Iterations**: 50, 100, 150

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/ann-smoking-classification.git
cd ann-smoking-classification

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset

Place the `smoking.csv` file in the `data/` directory. The dataset can be obtained from [Kaggle](https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset).

---

## Usage

### Jupyter Notebook

Run the full analysis notebook:

```bash
jupyter notebook smoking_classification.ipynb
```

### GUI Application

Launch the interactive Tkinter GUI:

```bash
python gui.py
```

The GUI allows you to:

- Load the dataset
- Adjust model parameters (hidden layers, learning rate, iterations)
- Run four evaluation scenarios
- View confusion matrices and network architecture diagrams
- Compare scenario performance

---

## Experimental Scenarios

| Scenario | Description | Validation |
|----------|-------------|------------|
| 1 | Training = Test | Same data for training and evaluation |
| 2 | 5-Fold Cross Validation | 5-fold stratified CV |
| 3 | 10-Fold Cross Validation | 10-fold stratified CV |
| 4 | 75-25 Split | 5 different random seeds (42, 123, 456, 789, 999) |

---

## Classification Results

Representative accuracy across scenarios (optimal architecture):

| Scenario | Accuracy |
|----------|----------|
| **1. Training=Test** | ~70.88% |
| **2. 5-Fold CV** | ~72.11% |
| **3. 10-Fold CV** | ~71.94% |
| **4. 75-25 Split** | ~71.82% |

**Note**: Scenario 1 may overfit; cross-validation yields more realistic performance estimates.

---

## Project Structure

```
.
├── data/
│   └── smoking.csv          # Dataset (not included in repo)
├── smoking_classification.ipynb  # Main analysis notebook
├── gui.py                   # Tkinter GUI application
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── .gitignore
```

---

## License

This project is provided for educational and research purposes. Please ensure compliance with the Kaggle dataset license when using the data.
