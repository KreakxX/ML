import keras
from keras import layers
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf


# Dataset
dataset, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)  # loading Dataset from imdb
train_data, test_data = dataset['train'], dataset['test']

tokenizer = tf.keras.layers.TextVectorization(max_tokens=10000, output_sequence_length=200)

x_train, y_train, x_test, y_test 


# Model
model= keras.Sequential([
  layers.Embedding(input_dim=10000, output_dim=16, input_length=100),
  layers.Dense(16,activation="relu"),       # if input is negative returning 0 => not for linear connection -> better learning for complex connections
  layers.Dense(1,activation="Sigmoid")   # Binary only 0 or 1 
])

# Compiling

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


#Training

history = model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.1)


# Evaluation
test_loss, test_acc = model.evaluate(x_test, y_test)

