# movie-review
# Sentiment Analysis with LSTM using Keras and TensorFlow

This repository contains a Python script for sentiment analysis using a Long Short-Term Memory (LSTM) neural network. Sentiment analysis aims to determine the sentiment or emotional tone of a given text, in this case, movie reviews. The code utilizes the Keras library with a TensorFlow backend and the IMDB dataset.

## Purpose
The purpose of this code is to create and train a sentiment analysis model to classify movie reviews as either positive or negative. The code provides functions for encoding and decoding text, making predictions, and demonstrating the model's sentiment analysis capability.

## Dependencies
Before using this code, make sure you have the following dependencies installed:
- Keras
- TensorFlow
- NumPy

You can install these libraries using pip:

```shell
pip install keras tensorflow numpy
```

## Dataset
The IMDB dataset is used for training and testing the sentiment analysis model. The dataset consists of movie reviews labeled as positive or negative.

## Usage
1. Load the IMDB dataset and preprocess the data by padding or truncating the reviews to a maximum length (MAXLEN).
2. Create a sequential neural network model with an embedding layer, an LSTM layer, and a dense layer for binary classification (positive or negative sentiment).
3. Train the model on the training data for a specified number of epochs.
4. Use the `predict` function to make sentiment predictions on custom text.

## Code Structure
- `train_data` and `test_data`: Loaded and preprocessed movie reviews from the IMDB dataset.
- `model`: A sequential model consisting of an embedding layer, LSTM layer, and a dense layer for binary classification.
- `train_model()`: Compiles and trains the model on the training data.
- `encode_text(text)`: Encodes a text input for prediction.
- `decode_function(integers)`: Decodes an encoded sequence back into text.
- `predict(text)`: Predicts the sentiment (positive or negative) of input text.

## Example
The code demonstrates the sentiment analysis by predicting the sentiment of two sample movie reviews: one positive and one negative.

```python
positive_rev = "that movie was good, I would watch it again"
predict(positive_rev)

negative_rev = "this movie is an abomination, very bad movie"
predict(negative_rev)
```

## Acknowledgments
This code is adapted from various sources and examples in the Keras and TensorFlow documentation and tutorials.

Feel free to use, modify, and extend this code for your own sentiment analysis tasks or natural language processing projects. Happy sentiment analysis!
