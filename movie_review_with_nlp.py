# -*- coding: utf-8 -*-
"""movie review with NLP.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sIwqmHY3rBwh5_HMoC48S4y1abDZk2al

movie review dataset
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
from keras.datasets import imdb
from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np

VOCAB_SIZE=88584
MAXLEN=250
(train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words = VOCAB_SIZE)

#check a review
train_data[4]

#each review in the dataset has different length so we must preprocess by padding or flatten depending on their size
train_data = sequence.pad_sequences(train_data,MAXLEN)
test_data = sequence.pad_sequences(test_data,MAXLEN)

#create a model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE,32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1,activation="sigmoid")
])

model.summary()

#train the model
model.compile(
    loss="binary_crossentropy",
    optimizer="rmsprop",
    metrics=['acc']
)
history=model.fit(train_data,train_labels,epochs=10,validation_split=0.2)

#making predictions
word_index = imdb.get_word_index()

def encode_text(text):
  tokens = keras.preprocessing.text.text_to_word_sequence(text)
  tokens = [word_index[word] if word in word_index else 0 for word in tokens]
  return sequence.pad_sequences([tokens], MAXLEN)[0]
text = "that movie was just amazing, so amazing"
encoded = encode_text(text)
print(encoded)

#decode function by reversing the word index
reverse_word_index = {value: key for (key, value) in word_index.items()}

def decode_function(integers):
  PAD=0
  text=""
  for num in integers:
    if num!=0:
      text+=reverse_word_index[num] + " "
  return text[:1]
print(decode_function(encoded))

#prediction
def predict(text):
  encoded_text=encode_text(text)
  pred=np.zeros((1,250))
  pred[0]=encoded_text
  result=model.predict(pred)
  print(result)

positive_rev="that movie was good, i would watch it again"
predict(positive_rev)

negative_rev="this movie is an abomination, very bad movie"
predict(negative_rev)