# Kidney Stone Detection using CNN + Harris Hawks Optimization (HHO)

This project implements a deep learning system for automatic kidney stone detection from axial CT images. It compares a baseline Convolutional Neural Network (CNN) with a hybrid CNN optimized using the Harris Hawks Optimization (HHO) algorithm.

The goal is to evaluate whether metaheuristic optimization can improve CNN performance by automatically tuning hyperparameters and architecture choices.

---

## Project Overview

Kidney stones are commonly diagnosed using CT imaging. Manual analysis can be time-consuming and subject to human error. This project explores how machine learning can assist medical imaging diagnostics by automatically classifying CT images into:

- Stone
- Non-Stone

Two models are implemented and compared.

Baseline CNN
- Standard CNN architecture
- Manually defined hyperparameters

CNN + HHO
- Same CNN architecture
- Hyperparameters optimized using Harris Hawks Optimization

---

## Project Structure

project/
│
├── baseline_fixed.py
├── fixed_hho.py
├── model_analysis.py
│
├── best_fixed_model.pt
├── hho_cnn_presentation_best.pt
│
├── training_results.csv
├── comparison_avg_metrics_summary.csv
├── comparison_per_class_metrics.csv
│
├── requirements.txt
│
└── Dataset/
    ├── Stone/
    └── Non-Stone/

---

## Dataset

Dataset: Axial CT Imaging Dataset for AI-Powered Kidney Stone Detection

Source:
https://www.kaggle.com/datasets/shuvokumarbasakbd/kidney-stone-axial-ct-imaging-colorized-dataset

The dataset contains CT images labeled as:

- Stone
- Non-Stone

Expected dataset structure:

Dataset/
├── Stone/
│   ├── image_001.jpg
│   └── ...
│
└── Non-Stone/
    ├── image_001.jpg
    └── ...

---

## Installation

Create a virtual environment:

python -m venv venv

Activate environment

Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Or manually:

pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn

---

## Dataset Setup

After downloading the dataset:

1. Place the dataset folder inside the project directory or anywhere on your system.

2. Update the dataset path in the training scripts.

Example:

data_dir = "./path/to/Dataset"

Make sure the folder contains:

Stone/
Non-Stone/

---

## How to Run

Train Baseline CNN

python baseline_fixed.py

This will:
- Train the CNN model
- Run for 15 epochs
- Save the best model

Outputs:

runs_baseline/
    best_model.pt
    history.csv

---

Train CNN + HHO Model

python fixed_hho.py

Quick test run:

python fixed_hho.py --quick

Skip optimization:

python fixed_hho.py --skip_hho

Outputs:

hho_cnn_presentation_best.pt
training_results.csv

---

Generate Model Comparison Plots

python model_analysis.py

This script generates:

- Confusion matrices
- Accuracy / Precision / Recall / F1 curves
- Model comparison charts
- Per-class performance metrics

Outputs are saved in:

comparison_outputs/

---

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Training vs Validation Curves
- Per-class performance comparison

---

## Harris Hawks Optimization (HHO)

HHO is a swarm-based metaheuristic optimization algorithm inspired by the cooperative hunting behavior of Harris hawks.

In this project HHO automatically optimizes:

- Learning rate
- Weight decay
- Dropout rate
- Batch size
- Data augmentation strength
- CNN channel scaling
- Optional network depth

This allows the model to search for better hyperparameters than manual tuning.

---

## Hardware

Recommended:

- GPU (CUDA)
or
- Apple Silicon (MPS)

CPU training is possible but slower.
