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
