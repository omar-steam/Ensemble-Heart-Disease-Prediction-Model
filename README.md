# Ensemble-Heart-Disease-Prediction-Model
This project aims to predict heart disease using various machine learning techniques, including K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and Neural Networks. It also combines these models into an ensemble for improved prediction accuracy.

The dataset used is sourced from the UCI Machine Learning Repository's Heart Disease dataset. The project includes preprocessing steps, training individual models, and creating an ensemble model that mixes KNN, SVM, and Neural Networks.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Data](#data)
5. [Models](#models)
6. [Evaluation](#evaluation)
7. [Ensemble Model](#ensemble-model)

## Project Structure

```bash
Heart-Disease-Prediction/
│
├── README.md          # Project documentation
├── requirements.txt   # Python dependencies
├── data/              # Dataset folder
│   └── heart.csv      # Heart disease dataset
├── notebooks/         # Jupyter notebooks for experiments and visualization
│   └── Heart_Disease_Prediction.ipynb # Step-by-step notebook
├── src/               # Source code for different parts of the project
│   ├── preprocessing.py  # Data preprocessing functions
│   ├── knn.py            # KNN model training
│   ├── svm.py            # SVM model training
│   ├── neural_network.py  # Neural network training
│   ├── ensemble.py       # Ensemble model combining KNN, SVM, and Neural Network
└── images/            # Images generated during the project (plots, heatmaps)
    └── heatmap.png    # Correlation heatmap of features
    └── accuracy.png   # Accuracy comparison of models
```
## Installation

1. Clone the repository:
```bash
git clone https://github.com/omar-steam/Ensemble-Heart-Disease-Prediction-Model
```






