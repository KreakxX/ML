import tensorflow as tf
import keras 
from keras import models,layers
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import os
input_shape = (256,256,3)

filepath = r"C:\Users\Henri\Downloads\FacialEmotion\data"

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,    # neccessary if the data same is
    zoom_range=0.1, 
    brightness_range=[0.8,1.2],
    horizontal_flip=False  
)

# training Dataset
train_generator = datagen.flow_from_directory(
    filepath,
    target_size=(256, 256),   # Shape Input size
    batch_size=16,
    class_mode='categorical',  # for more classes
    subset='training',
    shuffle=True
)

# validation Dataset
val_generator = datagen.flow_from_directory(
    filepath,
    target_size=(256, 256),
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

class_names = [d for d in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, d))]
class_names_length = len(class_names)

classification_model  = keras.Sequential([
  layers.Input(shape=input_shape),
  layers.Conv2D(32,3,activation="relu"),
  layers.MaxPooling2D(),
  layers.Conv2D(64,3,activation="relu"),
  layers.MaxPooling2D(),
  layers.Conv2D(128,3,activation="relu"),
  layers.MaxPooling2D(),
  layers.Conv2D(256,3,activation="relu"),
  layers.MaxPooling2D(),
  layers.Conv2D(512,3,activation="relu"),
  layers.GlobalAveragePooling2D(),
  layers.Dense(class_names_length,activation="softmax")
])

classification_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']   
)


earlyStopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights= True
)

classification_model.fit(train_generator, validation_data=val_generator,epochs=100,callbacks=[earlyStopping])
classification_model.save("EmotionClassifier2.keras")
