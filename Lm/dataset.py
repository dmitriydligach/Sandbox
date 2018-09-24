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
sys.dont_write_bytecode = True
import configparser, collections, time, nltk

class DatasetProvider:
  """Corpus for training a word-based language model"""

  def __init__(
    self,
    path,
    step,
    seq_len,
    min_tf=1,
    max_tokens=None):
    """Index words by frequency in a file"""

    self.path = path
    self.step = step
    self.seq_len = seq_len
    self.min_tf = min_tf
    self.max_tokens = max_tokens

    self.txt_as_ints = [] # entire corpus as seq of ints
    self.token2int = {}   # words indexed by frequency
    self.int2token = {}   # reverse index

    self.make_alphabet()
    self.text_to_int_seq()

  def read_train_text(self):
    """Obtain corpus as list"""

    text = open(self.path).read().lower()
    return nltk.word_tokenize(text)

  def make_alphabet(self):
    """Map words to ints and back"""

    counts = collections.Counter(self.read_train_text())

    index = 2
    self.token2int['padding'] = 0
    self.token2int['oov_word'] = 1

    for token, count in counts.most_common(self.max_tokens):
      self.token2int[token] = index
      self.int2token[index] = token
      index = index + 1

      if count < self.min_tf:
        break

    ts = [t for i, t in self.int2token.items()]
    print('most frequent tokens:', ' '.join(ts[:50]), '\n')

  def text_to_int_seq(self):
    """Convert corpus to a list of integers"""

    for token in self.read_train_text():
      if token in self.token2int:
        self.txt_as_ints.append(self.token2int[token])
      else:
        # this shouldn't happen when training on all data
        self.txt_as_ints.append(self.token2int['oov_word'])

  def make_train_data(self):
    """Make xs and ys to train on"""

    x = [] # sequences of ints
    y = [] # targets

    for i in range(
               0,
               len(self.txt_as_ints) - self.seq_len,
               self.step):
      x.append(self.txt_as_ints[i: i + self.seq_len])
      y.append(self.txt_as_ints[i + self.seq_len])

    return np.array(x), np.array(y)

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  corpus = cfg.get('args', 'corpus')
  step = cfg.getint('args', 'step')
  seqlen = cfg.getint('args', 'seqlen')

  t0 = time.time()
  datprov = DatasetProvider(corpus, step, seqlen)
  t1 = time.time()
  print('alphabet time:', t1 - t0)

  x, y = datprov.make_train_data()
