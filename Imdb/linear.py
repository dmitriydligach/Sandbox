#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True

import os, configparser

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import imdb

def main():
  """Fine-tune bert"""

  train_data = imdb.ImdbData(
    os.path.join(base, cfg.get('data', 'dir_path')),
    partition='train',
    n_files=cfg.get('data', 'n_files'))
  tr_texts, tr_labels = train_data.read()

  val_data = imdb.ImdbData(
    os.path.join(base, cfg.get('data', 'dir_path')),
    partition='test',
    n_files=cfg.get('data', 'n_files'))
  val_texts, val_labels = val_data.read()

  print('loaded %d training and %d validation samples' % \
        (len(tr_texts), len(val_texts)))

  vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=cfg.getint('model', 'num_features'),
    ngram_range=(1, 1))

  tr_texts = vectorizer.fit_transform(tr_texts)
  val_texts = vectorizer.transform(val_texts)

  classifier = LogisticRegression(C=1)
  classifier.fit(tr_texts, tr_labels)
  predictions = classifier.predict(val_texts)

  acc = accuracy_score(val_labels, predictions)
  print('accuracy = {}'.format(acc))

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  main()
