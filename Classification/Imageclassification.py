import keras 
from keras import layers,models
import tensorflow as tf

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

# load Dataset



#model.fit(train_ds,epochs=10)
model.save("Cat_Dog_Classifier.keras")