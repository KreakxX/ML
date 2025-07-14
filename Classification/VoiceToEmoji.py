import tensorflow as tf
import keras
from keras import models,layers
import librosa
import numpy as np




# input shape same as the audio_feature vector = 40
input_shape = (40,)

# basic classification model
AudioClassificationModel = keras.Sequential([
  layers.Input(shape=input_shape),
  layers.Dense(64,activation="relu"),
  layers.Dropout(0.3),
  layers.BatchNormalization(),

  layers.Dense(128,activation="relu"),
  layers.Dropout(0.3),
  layers.BatchNormalization(),

  layers.Dense(256,activation="relu"),
  layers.Dropout(0.3),

  layers.Dense(3,activation="softmax")
])


# compiling model
AudioClassificationModel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

y, sr = librosa.load("audio.wav", duration=3, offset=0.5)
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
mfccs = np.mean(mfccs.T, axis=0) 

X_train = []
y_train = []

AudioClassificationModel.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
