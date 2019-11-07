#!/usr/bin/env python3

import torch, glob, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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

class ImdbDataset(Dataset):
  """Used to fetch individual vectorized examples"""

  def __init__(self):
    """Constructor"""

    self.vectorize()

  def load_data(self):
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

  def vectorize(self):
    """Vectorize IMDB sentences"""

    sentences, labels = self.load_data()

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

    self.x_train = torch.tensor(x_train.toarray()).float()
    self.y_train = torch.tensor(y_train).float()
    self.x_dev = torch.tensor(x_dev.toarray()).float()
    self.y_dev = torch.tensor(y_dev).float()

  def get_dev(self):
    """Return the development set"""

    return self.x_dev, self.y_dev

  def __len__(self):
    """Should return the size of the datataset"""

    return len(self.x_train)

  def __getitem__(self, index):
    """Primary entry point for PyTorch datasets"""

    return self.x_train[index], self.y_train[index]

class Perceptron(nn.Module):
  """A Perceptron is one Linear layer"""

  def __init__(self, input_dim):
      """Args: input_dim (int): size of the input features"""

      super(Perceptron, self).__init__()
      self.fc1 = nn.Linear(input_dim, 1)

  def forward(self, x):
    """x.shape should be (batch, input_dim)"""

    return torch.sigmoid(self.fc1(x))

def train():
  """Training loop"""

  perceptron = Perceptron(input_dim=max_features)
  optimizer = optim.Adam(params=perceptron.parameters(), lr=lr)
  bce_loss = nn.BCELoss()

  dataset = ImdbDataset()
  # alternatively can use TensorDataset(x_train, y_train)
  # and then pass it to DataLoader(...) to make batches

  x_dev, y_dev = dataset.get_dev()
  batch_generator = DataLoader(dataset=dataset, batch_size=batch_size)

  for epoch in range(epochs):

    for x_train, y_train in batch_generator:

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

  train()
