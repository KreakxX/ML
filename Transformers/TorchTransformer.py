import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow_datasets as tfds

dataset, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)  # loading Dataset from imdb


def extractLabels(dataset):
  texts = []
  labels = []
  for text, label in tfds.as_numpy(dataset):
    texts.append(text)
    labels.append(label)
  return texts,labels

# tokenizer, rnn with attention layer (attention scores)

