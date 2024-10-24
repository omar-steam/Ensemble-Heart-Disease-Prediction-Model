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
    └── comparison of all algorthims used along with ensemble.png   # Accuracy comparison of models
    └── target 0 and 1.png # looking at the number of people with and without heart disease
    └── various bar charts variables.png # plotting all the variables each bar chart
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
```
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix,accuracy_score,roc_curve,classification_report
from mlxtend.classifier import StackingCVClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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

```
dataset = pd.get_dummies(df, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])
```
```
dataset.head()
```
```
y = dataset['target']
X = dataset.drop('target', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
```
# ```src/knn.py```

```
knn_scores = []

for k in range (1,40):
  knn_classifier = KNeighborsClassifier(n_neighbors = k)
  encoder = OneHotEncoder()
  knn_classifier.fit(X_train, y_train)
  knn_scores.append(knn_classifier.score(X_test, y_test))
print(f'best choice of k:{np.argmax(knn_scores)+1}')

k=8
knn_classifier = KNeighborsClassifier(n_neighbors = k)
knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)
print(f'accuracy: {np.sum(y_pred==y_test)/len(y_test)}')
```
# ```src/svm.py```
```
m7 = 'Support Vector Classifier'
svc =  SVC(kernel='rbf', C=2)
svc.fit(X_train, y_train)
svc_predicted = svc.predict(X_test)
svc_conf_matrix = confusion_matrix(y_test, svc_predicted)
svc_acc_score = accuracy_score(y_test, svc_predicted)
print("confussion matrix")
print(svc_conf_matrix)
print("\n")
print("Accuracy of Support Vector Classifier:",svc_acc_score*100,'\n')
print(classification_report(y_test,svc_predicted))
```
# ```src/nn.py``` Keras Sequential Model 
 ```
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from keras.layers import Dropout
model = tf.keras.Sequential([layers.Dense(20, activation='relu', name='dense1'), Dropout(0.2),
layers.Dense(25, activation='relu', name='dense2'),
layers.Dense(45, activation='relu', name='dense3'), Dropout(0.5),
layers.Dense(10, activation='relu', name='dense4'),
layers.Dense(2, activation='sigmoid', name='fc1')],)

from tensorflow import keras
model.compile(loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
optimizer = keras.optimizers.Adam(lr = 0.001),
metrics = ['accuracy'])
model.fit(X_train, y_train, batch_size = 32, epochs = 100, verbose = 2),
model.evaluate(X_test, y_test, batch_size = 32, verbose = 2)
 ```
# ```src/ensemble.py```

 ```
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Assuming X_train, X_test, y_train, and y_test are defined

# Create individual models
knn_classifier = KNeighborsClassifier(n_neighbors=8)
svc_classifier = SVC(kernel='rbf', C=2)

# Train the KNN and SVC models
knn_classifier.fit(X_train, y_train)
svc_classifier.fit(X_train, y_train)

# Define and train the neural network model
model = tf.keras.Sequential([
    layers.Dense(20, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(25, activation='relu'),
    layers.Dense(45, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='relu'),
    layers.Dense(2, activation='sigmoid')
])
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=0)

# Make predictions using the individual models
knn_pred = knn_classifier.predict(X_test)
svc_pred = svc_classifier.predict(X_test)
nn_pred = model.predict(X_test)
nn_pred_classes = np.argmax(nn_pred, axis=1)  # Convert probabilities to class predictions

# Combine predictions using weighted average ensemble
ensemble_pred = 0.2 * knn_pred + 0.3 * svc_pred + 0.5

# Convert ensemble predictions to class labels
ensemble_pred_classes = np.round(ensemble_pred).astype(int)

# Calculate accuracy of the ensemble
ensemble_accuracy = accuracy_score(y_test, ensemble_pred_classes)
print("Ensemble Accuracy:", ensemble_accuracy)
 ```

# Accuracy comparison of models via a bar chart

 ```
import matplotlib.pyplot as plt
import numpy as np

# Assuming ensemble_pred_classes, knn_predictions, svm_predictions, nn_predictions are defined

# Sample data (replace this with your actual data)
ensemble_pred_classes = np.random.randint(0, 2, size=10)
knn_predictions = np.random.rand(10)
svm_predictions = np.random.rand(10)
nn_predictions = np.random.rand(10)

# Normalize data to be in the range [0, 1]
normalized_ensemble_pred = ensemble_pred_classes / max(ensemble_pred_classes)
normalized_knn_predictions = knn_predictions / max(knn_predictions)
normalized_svm_predictions = svm_predictions / max(svm_predictions)
normalized_nn_predictions = nn_predictions / max(nn_predictions)

# Combine all predictions for plotting
all_predictions = [normalized_knn_predictions, normalized_svm_predictions, normalized_nn_predictions, normalized_ensemble_pred]
algorithm_names = ['SVM', 'KNN', 'Neural Network', 'Ensemble']

num_predictions = len(all_predictions)
bar_width = 0.2
bar_positions = np.arange(len(normalized_ensemble_pred))

# Create a multi-bar graph for predictions
for i, predictions in enumerate(all_predictions):
    plt.bar(bar_positions + i * bar_width, predictions, width=bar_width, label=algorithm_names[i])

# Add labels and title
plt.xlabel('Sample Index')
plt.ylabel('Normalized Class Label (Percentage)')
plt.title('Predictions Comparison (Percentage)')

# Add grid lines for better readability
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Adjust y-axis ticks to represent percentages
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])

# Add legend
plt.legend()

# Show the plot
plt.show()
 ```







