import keras 
from keras import layers,models
import tensorflow as tf
import os
from PIL import Image
import tensorflow_datasets as tfds
import numpy as np


# CNN Convulational Neural Network
model = keras.Sequential(
  [layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),   # 32 filters like finding shapes and corners that,  3,3 is for the size like a 3x3 pixel square  if smaller = feiner
    layers.MaxPooling2D(2,2), # makes the image size / 2 -> performance optimization 64x64  takes out of each 2x2 sqaure the max value so the sizes geht divided by 2
    layers.Conv2D(64, (3,3), activation='relu'),   # another one with more filters
    layers.MaxPooling2D(2,2), # performacnce 32x32
    layers.GlobalAveragePooling2D(),   # prevents overfitting and makes it more stable
    layers.Dense(64, activation='relu'),  # neurons             # could implement dropout for more stability and prevent overfitting
    layers.Dense(2, activation='softmax')   # output two classes as binary 0 1 for cat or dog
]
)

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy", metrics=["accuracy"] ) # compile

# load Dataset 
(train_ds, test_ds), info = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:]'],
    shuffle_files=True, 
    as_supervised=True,
    with_info=True
)
classnames = info.features['label'].names

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # normalization 
    image = tf.image.resize(image, [128, 128]) # and rezising for the input to match
    return image, label

# Make Data in different kinds so he learns better, and prevents overfitting             # could make more features to generalise it better
def dataAugmentaion(image,label):
    image = tf.image.random_flip_left_right(image)          # and rotation 
    image = tf.image.random_brightness(image, 0.2)              # random brightness          -> prevents overfitting
    image = tf.image.random_contrast(image, 0.8, 1.2)           # random contrast
    return image, label
    
train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.map(dataAugmentaion, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.map(preprocess,num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.map(dataAugmentaion,num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(32).prefetch(tf.data.AUTOTUNE)


# early_stopping = keras.callbacks.EarlyStopping(
#     monitor='val_loss',                             # early stopping to prevent overfitting
#     patience=4,              
#     restore_best_weights=True  
# )

# model.fit(
#     train_ds,
#     validation_data=test_ds,
#     epochs=60,                       #training
#     callbacks=early_stopping
# )
# model.save("Cat_Dog_Classifier3.keras")


def CheckModelPerformance(modelName, ingputImg):
    global classnames
    model = keras.models.load_model(modelName)
    img = tf.keras.utils.load_img(ingputImg, target_size=(128, 128))  # load image with the target size
    img_array = tf.keras.utils.img_to_array(img)  # loading into numpy array because is neccessary for the model
    img_array = img_array/255.0          # normalization value between 0 and 1
    img_array = np.expand_dims(img_array,axis=0)          # add batch dimension with numpy

    prediction = model.predict(img_array)        # returns np array with multiple values but we need the highest like 0 or 1 
    confidence = np.max(prediction)
    predicted_class = np.argmax(prediction)
    print(classnames[predicted_class])
    print(f"Confidence: {confidence}")


CheckModelPerformance("Cat_Dog_Classifier3.keras","Hund.jpg")        
    