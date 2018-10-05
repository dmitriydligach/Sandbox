#!/usr/bin/env python3

# reproducible results
import numpy as np
import random as rn
import tensorflow as tf
np.random.seed(1337)
rn.seed(1337)
tf.set_random_seed(1337)
import os, os.path
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = '0'
from keras import backend as bke
s = tf.Session(graph=tf.get_default_graph())
bke.set_session(s)

import sys
sys.dont_write_bytecode = True
import configparser, collections, time, nltk

XFILE = 'x.txt'
YFILE = 'y.txt'

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

    if not os.path.isfile(XFILE):
      self.train_to_int_seq()
      self.make_and_save_train_data()

  def read_train_text(self, use_tokenizer=False):
    """Obtain corpus as list. Split on spaces by default"""

    # open(...).read() fails on large files
    # assume entire file is one line for now
    text = open(self.path).readline().lower()
    print('done reading text...')
    if use_tokenizer:
      tokenized = nltk.word_tokenize(text)
      print('done tokenizing...')
      return tokenized
    else:
      tokenized = text.split()
      print('done tokenizing...')
      return tokenized

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
    print('most frequent tokens:', ' '.join(ts[:15]))
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
    print('integer sequence length:', len(self.txt_as_ints))

  def text_to_int_seq(self, text):
    """Convert a text fragment to a list of integers"""

    int_seq = []
    for token in nltk.word_tokenize(text.lower()):
      if token in self.token2int:
        int_seq.append(self.token2int[token])
      else:
        int_seq.append(self.token2int['oov_word'])

    return int_seq

  def int_seq_to_text(self, int_seq):
    """Convert a seq of ints to text"""

    txt = [self.int2token[i] for i in int_seq]
    return ' '.join(txt)

  def make_train_data(self):
    """Make xs and ys in memory to train on"""

    x = [] # sequences of ints
    y = [] # targets

    size = len(self.txt_as_ints) / float(self.step)
    print('making %d training examples' % size)

    for i in range(
               0,
               len(self.txt_as_ints) - self.seq_len,
               self.step):

      if len(x) % 25000000 == 0 and len(x) > 0:
        print('made 25M; total now:', len(x))

      x.append(self.txt_as_ints[i: i + self.seq_len])
      y.append(self.txt_as_ints[i + self.seq_len])

    print('done making training data...')
    return np.array(x), np.array(y)

  def make_and_save_train_data(self):
    """Make training data and save in file"""

    x_out = open(XFILE, 'w')
    y_out = open(YFILE, 'w')

    size = len(self.txt_as_ints) / float(self.step)
    print('making %d training examples' % size)

    for i in range(
               0,
               len(self.txt_as_ints) - self.seq_len,
               self.step):
      x = self.txt_as_ints[i: i + self.seq_len]
      y = self.txt_as_ints[i + self.seq_len]
      x_out.write('%s\n' % ' '.join(map(str, x)))
      y_out.write('%d\n' % y)

    print('done making training data...')
    x_out.close()
    y_out.close()

  def read_train_data_from_file(self, batch=50000):
    """Generator to read training data in batches"""

    line_num = 0

    x_batch = []
    y_batch = []

    for x_line, y_line in zip(open(XFILE), open(YFILE)):

      x = list(map(int, x_line.split()))
      y = int(y_line.strip())
      x_batch.append(x)
      y_batch.append(y)

      line_num += 1
      if line_num % batch == 0:
        yield np.array(x_batch), np.array(y_batch)
        print('fetched %d lines total...' % line_num)
        x_batch, y_batch = [], []

    # fetch remaining data
    yield np.array(x_batch), np.array(y_batch)
    print('fetched %d lines total...' % line_num)

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
