#!/usr/bin/env python3

import torch, numpy
import glob, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data_pos = 'Imdb/train/pos/*.txt'
data_neg = 'Imdb/train/neg/*.txt'

base = os.environ['DATA_ROOT']
data_pos = os.path.join(base, data_pos)
data_neg = os.path.join(base, data_neg)

# settings
gpu_num = 0
max_files = 500
batch_size = 8
epochs = 2

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

def logistic_regression():
  """Train a logistic regression classifier"""

  sentences, labels = load_data()

  x_train, x_test, y_train, y_test = \
    train_test_split(
      sentences,
      labels,
      test_size=0.1,
      random_state=0)

  vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000,
    ngram_range=(1, 1))

  x_train = vectorizer.fit_transform(x_train)
  x_test = vectorizer.transform(x_test)

  classifier = LogisticRegression(C=1, solver='liblinear')
  model = classifier.fit(x_train, y_train)
  predictions = classifier.predict(x_test)

  acc = accuracy_score(y_test, predictions)
  print('accuracy (test) = {}'.format(acc))

if __name__ == "__main__":

  logistic_regression()
