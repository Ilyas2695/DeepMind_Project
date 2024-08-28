# MLIR Cost Estimation

This project aims to estimate the computational cost of operations within MLIR (Multi-Level Intermediate Representation) files using a neural network model. By parsing MLIR files, counting operations, and utilizing a trained model, the project provides an estimated runtime cost for new MLIR files.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [Results](#results)
- [Potential Improvements](#potential-improvements)
- [Acknowledgments](#acknowledgments)

## Introduction

The MLIR Cost Estimation Project is designed to help developers predict the computational cost of MLIR files by analyzing the operations within them. The project uses a neural network to learn from empirical data collected from existing MLIR files, enabling the prediction of costs for unseen files.

## Features

- **Data Collection**: Parses MLIR files, counts operations, and calculates estimated costs.
- **Model Training**: Trains a neural network to predict operation costs based on operation counts.
- **Cost Prediction**: Predicts the computational cost of new MLIR files using the trained model.
- **Visualization**: Provides training and validation loss plots to assess model performance.

## Setup

### Prerequisites

- Python 3.8 or higher
- [TensorFlow](https://www.tensorflow.org/install) (for neural network modeling)
- [scikit-learn](https://scikit-learn.org/stable/install.html) (for data preprocessing)
- [Matplotlib](https://matplotlib.org/stable/users/installing.html) (for plotting)
- Required Python packages can be installed via pip:

    ```bash
    pip install tensorflow scikit-learn matplotlib
    ```

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/mlir-cost-estimation.git
    cd mlir-cost-estimation
    ```

2. Ensure all required packages are installed:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Collect Data
Run the data collection script to parse MLIR files and generate operation cost data:

```bash
python data_collection.py
```
This script reads MLIR files from a specified directory, counts operations, and saves the results in operation_costs.jsonl.

### Train the Model
Train the neural network model using the collected data:

```bash
python train_model.py
```
This will train the model, evaluate it on a test set, and save the trained model as cost_estimation_model.h5.

### Predict Costs
To predict the computational cost of new MLIR files, run:

```bash
python predict_cost.py
```
This script loads the trained model and predicts the cost for a specified MLIR file.

## Results
The training and validation loss graph provides insights into the model's performance over epochs:

