import keras 
from keras import layers,models
import tensorflow as tf
import os
from PIL import Image
import tensorflow_datasets as tfds

model = keras.Sequential(
  [layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),   # 32 filters like finding shapes and corners that,  3,3 is for the size if smaller = feiner
    layers.MaxPooling2D(2,2), # makes the image size / 2 -> performance optimization 32x32
    layers.Conv2D(64, (3,3), activation='relu'),   # another one with more filters
    layers.MaxPooling2D(2,2), # performacnce 16x16
    layers.GlobalAveragePooling2D(),   # prevents overfitting and makes it more stable
    layers.Dense(64, activation='relu'),  # neurons
    layers.Dense(2, activation='softmax')   # output two classes as binary 0 1 for cat or dog
]
)

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy", metrics=["accuracy"] ) # compile

# load Dataset  # data augmentation but not here because dataset is ideal

(train_ds, test_ds), info = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, [128, 128])
    return image, label

train_ds = train_ds.map(preprocess).batch(32)
test_ds = test_ds.map(preprocess).batch(32)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',      
    patience=3,              
    restore_best_weights=True  
)
model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=10,
    callbacks=early_stopping
)
model.save("Cat_Dog_Classifier.keras")
