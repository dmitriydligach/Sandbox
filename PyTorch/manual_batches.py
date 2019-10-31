#!/usr/bin/env python3

import torch, glob, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import numpy as np
import torch.nn as nn
import torch.optim as optim

data_pos = 'Imdb/train/pos/*.txt'
data_neg = 'Imdb/train/neg/*.txt'

base = os.environ['DATA_ROOT']
data_pos = os.path.join(base, data_pos)
data_neg = os.path.join(base, data_neg)

# settings
max_files = 1000
max_features = 5000

# hyper-parameters
lr = 0.01
batch_size = 100
epochs = 10

def load_data():
  """Rotten tomatoes"""

  labels = []
  sentences = []

  for file in glob.glob(data_pos)[:max_files]:
    labels.append(1)
    sentences.append(open(file).read().rstrip())
  for file in glob.glob(data_neg)[:max_files]:
    labels.append(0)
    sentences.append(open(file).read())

  return sentences, labels

def make_vectors():
  """Vectorize input sentences"""

  sentences, labels = load_data()

  x_train, x_dev, y_train, y_dev = \
    train_test_split(
      sentences,
      labels,
      test_size=0.1,
      random_state=0)

  vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=max_features,
    ngram_range=(1, 1))

  x_train = vectorizer.fit_transform(x_train)
  x_dev = vectorizer.transform(x_dev)

  # the output of transform() is a sparse matrix!
  return x_train, x_dev, y_train, y_dev

def logistic_regression():
  """Train a logistic regression classifier"""

  x_train, x_dev, y_train, y_dev = make_vectors()

  classifier = LogisticRegression(C=1, solver='liblinear')
  model = classifier.fit(x_train, y_train)
  predictions = classifier.predict(x_dev)

  acc = accuracy_score(y_dev, predictions)
  print('accuracy (dev) = {}'.format(acc))

def stream_data(batch_size):
  """Only train data. Dev stays the same"""

  x_train, x_dev, y_train, y_dev = make_vectors()
  x_dev = torch.tensor(x_dev.toarray()).float()

  for row in range(0, x_train.shape[0], batch_size):
    batch_x_train = x_train[row:row+batch_size, :]
    batch_y_train = y_train[row:row+batch_size]

    batch_x_train = torch.tensor(batch_x_train.toarray()).float()
    batch_y_train = torch.tensor(batch_y_train).float()

    yield batch_x_train, x_dev, batch_y_train, y_dev

class Perceptron(nn.Module):
  """A Perceptron is one Linear layer"""

  def __init__(self, input_dim):
      """Args: input_dim (int): size of the input features"""

      super(Perceptron, self).__init__()
      self.fc1 = nn.Linear(input_dim, 1)

  def forward(self, x):
    """x.shape should be (batch, input_dim)"""

    return torch.sigmoid(self.fc1(x))

def train_manual_batches():
  """Training loop"""

  perceptron = Perceptron(input_dim=max_features)
  optimizer = optim.Adam(params=perceptron.parameters(), lr=lr)
  bce_loss = nn.BCELoss()

  # do several passess over training examples
  for epoch in range(epochs):

    # do one pass over training examples
    for x_train, x_dev, y_train, y_dev in stream_data(batch_size):
      optimizer.zero_grad()

      y_predicted = perceptron(x_train).squeeze()
      loss = bce_loss(y_predicted, y_train)

      loss.backward()
      optimizer.step()

    predictions = perceptron(x_dev).squeeze()
    predictions = predictions > 0.5
    acc = accuracy_score(y_dev, predictions.tolist())

    print('ep: {}, loss: {}, acc: {}'.format(epoch+1, loss.item(), acc))

def train_no_batches():
  """Training loop"""

  perceptron = Perceptron(input_dim=max_features)
  optimizer = optim.Adam(params=perceptron.parameters(), lr=lr)
  bce_loss = nn.BCELoss()

  x_train, x_dev, y_train, y_dev = make_vectors()
  x_train = torch.tensor(x_train.toarray()).float()
  y_train = torch.tensor(y_train).float()
  x_dev = torch.tensor(x_dev.toarray()).float()
  y_dev = torch.tensor(y_dev).float()

  for epoch in range(epochs):
    optimizer.zero_grad()
    y_hat = perceptron(x_train).squeeze()
    loss = bce_loss(y_hat, y_train)
    loss.backward()
    optimizer.step()

    predictions = perceptron(x_dev).squeeze()
    predictions = predictions > 0.5
    acc = accuracy_score(y_dev, predictions.tolist())

    print('ep: {}, loss: {}, acc: {}'.format(epoch+1, loss.item(), acc))

if __name__ == "__main__":

  train_no_batches()
