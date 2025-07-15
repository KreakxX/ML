import tensorflow as tf
import keras
from keras import models,layers
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# input shape same as the audio_feature vector = 40
input_shape = (20,)

# basic classification model
AudioClassificationModel = keras.Sequential([
  layers.Input(shape=input_shape),
  layers.Dense(64,activation="relu"),
  layers.Dropout(0.3),    # prevent overfitting
  layers.BatchNormalization(),  # normalizing the values for better performance

  layers.Dense(128,activation="relu"),
  layers.Dropout(0.3),  # prevent overfitting
  layers.BatchNormalization(), # normalizing the values for better performance

  layers.Dense(256,activation="relu"),  
  layers.Dropout(0.3),  # prevent overfitting

  layers.Dense(10,activation="softmax")    # 10 Outputs 1 for each Genre
])  

# compiling model
AudioClassificationModel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # example data for testing
# y, sr = librosa.load("audio.wav", duration=3, offset=0.5)
# mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
# mfccs = np.mean(mfccs.T, axis=0) 

# load csv file
df = pd.read_csv(r"C:\Users\Henri\Videos\MusicGenreDataset\features_30_sec.csv")

# extract all features all 41 mfcc features
feature_cols = [f"mfcc{i}_mean" for i in range(1, 21)]

X = df[feature_cols].values  # all the mfcc values

le = LabelEncoder()
y = df['label'].values     # the genre
y = le.fit_transform(y)

# splitting the data in training and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

Earlystopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',      
    patience=20,              
    restore_best_weights=True  
)
# training model
AudioClassificationModel.fit(
    X_train, y_train,
    epochs=200,      # training for 20 epochs
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[Earlystopping]
)

AudioClassificationModel.save("models/AudioClassification.keras")
