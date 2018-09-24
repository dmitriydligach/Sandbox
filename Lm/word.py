#!/usr/bin/env python3

# reproducible results
import numpy as np
import random as rn
import tensorflow as tf
np.random.seed(1337)
rn.seed(1337)
tf.set_random_seed(1337)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = '0'
from keras import backend as bke
s = tf.Session(graph=tf.get_default_graph())
bke.set_session(s)

import sys
sys.path.append('../Lib/')
sys.dont_write_bytecode = True

import sklearn as sk
from sklearn.metrics import f1_score
import keras as k
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.embeddings import Embedding
from keras.layers import LSTM
import configparser
from dataset import DatasetProvider

def print_config():
  """Print config file for content"""

  for key, value in cfg.items('args'):
    print(key, '-', value)
  print()

def get_model(vocab_size, max_seq_len, num_targets):
  """Model definition"""

  model = Sequential()

  model.add(Embedding(input_dim=vocab_size,
                      output_dim=300,
                      input_length=max_seq_len))
  model.add(LSTM(cfg.getint('nn', 'units')))
  model.add(Dense(num_targets, activation='softmax'))
  model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop')

  return model

def main():
  """Driver function"""

  corpus = os.path.join(
    os.environ['DATA_ROOT'],
    cfg.get('args', 'train'))
  step = cfg.getint('args', 'step')
  maxlen = cfg.getint('args', 'maxlen')

  dp = DatasetProvider(corpus, step, maxlen)
  x, y = dp.make_train_data()
  y = to_categorical(y, len(dp.token2int))
  print('x / y shapes:', x.shape, y.shape)

  model = get_model(len(dp.token2int), maxlen, len(dp.token2int))
  model.fit(x,
            y,
            epochs=cfg.getint('nn', 'epochs'),
            batch_size=cfg.getint('nn', 'batch'),
            verbose=1,
            validation_split=0.0)

if __name__ == "__main__":

  # settings file specified as command-line argument
  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  print_config()

  main()
