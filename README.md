# Ensemble-Heart-Disease-Prediction-Model
This project aims to predict heart disease using various machine learning techniques, including K-Nearest Neighbours (KNN), Support Vector Machines (SVM), and Neural Networks. It also combines these models into an ensemble for improved prediction accuracy.

The dataset used is sourced from the UCI (University of California Irvine) Machine Learning Repository's Heart Disease dataset. The project includes preprocessing steps, training individual models, and creating an ensemble model that mixes KNN, SVM, and Neural Networks.

## Table of Contents

1. [Project Structure](#project-structure)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Data](#data)
5. [The Models](#models)
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

## 2. Run the Scripts

You can also run individual scripts located in the ```src``` directory to train models and get predictions

- **Preprocess the data**:
```
python src/preprocessing.py
```
- **KNN (K-Nearest Neighbours) Model training**:

```
python src/KNN.py
```

- SVM (Support Vector Machines) Model training:

```
python src/svm.py
```

- **NN (neural networks) Model training**:

```
python src/nn.py
```

- **Ensemble Model running**:

```
python src/ensemble.py
```

# Data 

**The dataset used is the UCI Heart Disease Dataset**:

- **Source**: Janosi, Andras, Steinbrunn, William, Pfisterer, Matthias, and Detrano, Robert. (1988). Heart Disease. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/45/heart+disease
- **Attributes**: The dataset includes 13 features related to heart health (age, sex, cholesterol, etc.) and a target variable indicating the presence or absence of heart disease.

# The Models 

**The following models are implemented and compared in this project**:

**1. K-Nearest Neigbours**: A non-parametric, instance-based learning method that classifies a new sample based on the majority class of its nearest neighbors.
We tune the ```k``` parameter to find the optimal number of neighbors.

**2. Support Vector Machines**: A powerful classifier that works by finding the hyperplane that best separates the data into classes.
We use the Radial Basis Function (RBF) kernel and tune the ```C``` parameter.

**3. Neural Networks (Keras Sequential Model)**: A deep learning model consisting of multiple layers of neurons. We use a fully connected feedforward neural network with ReLU activation for hidden layers and Sigmoid activation for the output layer. The model is compiled using Sparse Categorical Crossentropy as the loss function.

**4. Ensemble Model**: We combine the predictions from KNN, SVM, and the Neural Network to create an ensemble using weighted averaging. The ensemble model aims to leverage the strengths of each individual model to improve overall prediction accuracy.

## Evaluation

We evaluate each model based on the following metrics:

- **Confusion Matrix**: To visualize the number of correct and incorrect predictions.
- **Accuracy Score**: The percentage of correct predictions.
- **Classification Report**: Precision, Recall, and F1-Score for each class.


Visualisations such as a heatmap of feature correlations and accuracy comparisons via a bar chart are included in the ```images/``` folder.

# Ensemble Model

The ensemble model takes a weighted average of predictions from KNN, SVM, and the Neural Network, and then rounds the result to determine the final class label. The weights for the models are as follows:

- **KNN**: 0.2
- **SVM**: 0.3
- **Neural Network**: 0.5

# Contributing

You can fix this repository and submit pull requests. Contributions are welcome!

# License
This project is licensed under the Apache License 2.0. See the ```LICENSE``` file for details.

### `requirements.txt`

```txt
numpy==1.23.4
pandas==1.5.2
scikit-learn==1.1.3
matplotlib==3.6.2
seaborn==0.12.1
tensorflow==2.10.0
mlxtend==0.19.0
```
# Importing Libraries and Setting Up the Environment

We import the necessary libraries and configure the environment to perform data manipulation, visualization, and machine learning tasks. These libraries include tools for numerical computations, data handling, plotting, and machine learning. Additionally, we suppress warnings to ensure a cleaner output.

```
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.cm import rainbow
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
```
# Loading up the dataset 

```
df = pd.read_csv('heart.csv')
```
# Creating the chart for the dataset 

```
df.info()

df.describe() 

import seaborn as sns
#get correlations of each features in dataset
corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
```
```
df.hist()
```
```
sns.set_style('whitegrid')
sns.countplot(x='target',data=df,palette='RdBu_r')
```

# ```src/preprocessing.py```















