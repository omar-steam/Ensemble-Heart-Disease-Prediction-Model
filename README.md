# Ensemble-Heart-Disease-Prediction-Model
This project aims to predict heart disease using various machine learning techniques, including K-Nearest Neighbours (KNN), Support Vector Machines (SVM), and Neural Networks. It also combines these models into an ensemble for improved prediction accuracy.

The dataset used is sourced from the UCI (University of California Irvine) Machine Learning Repository's Heart Disease dataset. The project includes preprocessing steps, training individual models, and creating an ensemble model that mixes KNN, SVM, and Neural Networks.

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
2. Install the required Python packages for this prediction model
```bash
pip install -r requirements.txt
```

## Usage 

## 1. Run the Project in Google Colab
To run this project in Google Colab, follow these steps:

1. **Open Google Colab**: Navigate to Google Colab.

2. **Upload the Notebook**:

Click on File → Upload Notebook.
Upload the notebook located at ```notebooks/Heart_Disease_Prediction.ipynb```.

3. **Upload the Dataset**:

After uploading the notebook, you will need to upload the heart.csv dataset.
Run the following command in the first cell of the notebook to upload the CSV file to Colab:
```
from google.colab import files
uploaded = files.upload()
```
After running the cell, use the dialogue box to upload the heart.csv file.

4. **Install Dependencies**: Google Colab comes pre-installed with most libraries. However, if any specific dependencies are missing, you can install them using the following commands in the notebook:
```
!pip install -r requirements.txt
```
This will install all required Python packages.

5. **Run the Notebook**: Once the dataset is uploaded and dependencies are installed, run the cells in the notebook sequentially to see the data preprocessing, model training, and evaluation steps.






