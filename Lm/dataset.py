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
    self.train_to_int_seq()

  def read_train_text(self, use_tokenizer=False):
    """Obtain corpus as list. Split on spaces by default"""

    # open(...).read() fails on large files
    # assume entire file is one line for now
    text = open(self.path).readline().lower()
    print('done reading text...')
    if use_tokenizer:
      return nltk.word_tokenize(text)
    else:
      return text.split()

  def make_alphabet(self):
    """Map words to ints and back"""

    counts = collections.Counter(self.read_train_text())
    print('done counting tokens...')

    index = 2
    self.token2int['padding'] = 0
    self.token2int['oov_word'] = 1
    self.int2token[0] = 'padding'
    self.int2token[1] = 'oov_word'

    for token, count in counts.most_common(self.max_tokens):
      self.token2int[token] = index
      self.int2token[index] = token
      index = index + 1
      if count < self.min_tf:
        break

    ts = [t for i, t in self.int2token.items()]
    print('most frequent tokens:', ' '.join(ts[:50]))
    print('vocabulary size:', len(self.token2int))

  def train_to_int_seq(self):
    """Convert training corpus to a list of integers"""

    print('converting text to integers...')
    for token in self.read_train_text():
      if token in self.token2int:
        self.txt_as_ints.append(self.token2int[token])
      else:
        # e.g. low freqency tokens
        self.txt_as_ints.append(self.token2int['oov_word'])
    print('done converting...')

  def text_to_int_seq(self, text):
    """Convert a text fragment to a list of integers"""

    int_seq = []
    for token in nltk.word_tokenize(text.lower()):
      if token in self.token2int:
        int_seq.append(self.token2int[token])
      else:
        int_seq.append(self.token2int['oov_word'])

    return int_seq

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
