import keras 
from keras import layers,models
import tensorflow as tf
import os
from PIL import Image
import tensorflow_datasets as tfds
import numpy as np
from datasets import load_dataset


# dataset preparation 

train_ds, test_ds,info = tfds.load('food101', split=['train', 'validation'], as_supervised=True,with_info=True)

# for better learning
def dataAugmentaion(image,label):
    image = tf.image.random_flip_left_right(image)          # and rotation 
    image = tf.image.random_brightness(image, 0.2)              # random brightness          -> prevents overfitting
    image = tf.image.random_contrast(image, 0.8, 1.2)           # random contrast
    return image, label

# data preparation
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0  
    image = tf.image.resize(image, [128, 128]) 
    return image, label

train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.map(dataAugmentaion, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess,num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.map(dataAugmentaion,num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(32).prefetch(tf.data.AUTOTUNE)

class_names = info.features["label"].names
class_names_length = len(class_names)

input_shape = (256,256,3)

# model preparation
model = keras.Sequential([
  layers.Input(shape=input_shape),
  layers.Conv2D(64,3,activation="relu"),
  layers.MaxPooling2D(),
  layers.Conv2D(128,3,activation="relu"),
  layers.MaxPooling2D(),
  layers.Conv2D(256,3,activation="relu"),
  layers.GlobalAveragePooling2D(),
  layers.Dense(128,activation="relu"),
  layers.Dense(class_names_length,activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
)

# training
earlyStopping = keras.callbacks.EarlyStopping(
  monitor="val_loss",
  patience=10,
  restore_best_weights=True
)

model.fit(train_ds, validation_data=test_ds,epochs=10,callbacks=[earlyStopping])
model.save("Food_classfier.keras")