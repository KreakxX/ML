import keras
from keras import layers
import tensorflow_datasets as tfds
import tensorflow as tf
from keras.callbacks import EarlyStopping


# Dataset
dataset, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)  # loading Dataset from imdb
train_data, test_data = dataset['train'], dataset['test']

tokenizer = tf.keras.layers.TextVectorization(max_tokens=10000, output_sequence_length=100) 
train_text = train_data.map(lambda text, label: text)
tokenizer.adapt(train_text) 


# Model no recurrent Network its just a basic nn
model= keras.Sequential([
  layers.Embedding(input_dim=10000, output_dim=16, input_length=100), 
  layers.GlobalAveragePooling1D(),
  layers.Dense(32,activation="relu"), 
  layers.Dense(32,activation="relu"),      # if input is negative returning 0 => not for linear connection -> better learning for complex connections
  layers.Dense(1,activation="sigmoid")   # Binary only 0 or 1 
])

# Training and Test Data
def tokenizeInput(text,label):
  return tokenizer(text),label


train_ds = train_data.map(tokenizeInput).batch(32).prefetch(tf.data.AUTOTUNE)
test_ds = test_data.map(tokenizeInput).batch(32).prefetch(tf.data.AUTOTUNE)


# Compiling
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


# Early stopping einbauen
early_stopping = EarlyStopping(
    monitor='val_loss',      
    patience=3,              
    restore_best_weights=True  
)
#Training
model.fit(train_ds, validation_data=test_ds, epochs=30, callbacks=[early_stopping])


# Evaluation
test_loss, test_acc = model.evaluate(test_ds)
