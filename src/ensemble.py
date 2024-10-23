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
