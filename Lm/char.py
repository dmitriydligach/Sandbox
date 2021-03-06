#!/usr/bin/env python3

# reproducible results
import numpy as np
import random as rn
import tensorflow as tf
np.random.seed(100)
rn.seed(100)
tf.set_random_seed(100)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = '0'
from keras import backend as bke
s = tf.Session(graph=tf.get_default_graph())
bke.set_session(s)

# the rest of the imports
import keras, random, sys, configparser
import hurry.filesize
from keras import layers

def get_corpus():
  """Raw corpus"""

  path = cfg.get('args', 'corpus')
  return open(path).read().lower()

def make_char_alphabet(text):
  """Map characters to integers"""

  chars = sorted(list(set(text)))
  return dict((char, chars.index(char)) for char in chars)

def make_training_data(text, char2int):
  """Make x and y"""

  seqlen = cfg.getint('args', 'seqlen')

  sequences = [] # sequences of seqlen characters
  targets = []   # char that follows each sequence above

  step = cfg.getint('args', 'step')
  for i in range(0, len(text) - seqlen, step):
      sequences.append(text[i: i + seqlen])
      targets.append(text[i + seqlen])

  # vectorize sequences; make a tensor of the following shape:
  # (samples, time_steps, features) -> (samples, seqlen, uniq_chars))

  items = len(sequences) * seqlen * len(char2int)
  item_size_in_bytes = np.dtype(np.bool).itemsize
  print('train tensor shape:', (len(sequences), seqlen, len(char2int)))
  print('allocating:', hurry.filesize.size(items * item_size_in_bytes))
  x = np.zeros((len(sequences), seqlen, len(char2int)), dtype=np.bool)
  print('train tensor size in bytes:', hurry.filesize.size(x.nbytes))
  y = np.zeros((len(sequences), len(char2int)), dtype=np.bool)

  for n, sequence in enumerate(sequences):
      for time_step, char in enumerate(sequence):
          x[n, time_step, char2int[char]] = 1
      y[n, char2int[targets[n]]] = 1

  return x, y

def get_model(num_features):
  """Model that takes (time_steps, features) as input"""

  model = keras.models.Sequential()
  model.add(layers.LSTM(128, input_shape=(None, num_features)))
  model.add(layers.Dense(num_features, activation='softmax'))
  optimizer = keras.optimizers.RMSprop(lr=0.01)
  model.compile(loss='categorical_crossentropy', optimizer=optimizer)

  return model

def sample_char(preds, temperature=1.0):
  """Reweight the prob distribution and sample a character"""

  # preds shape: (len(chars),)
  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)

  return np.argmax(probas)

def pick_init_seq(text):
  """Pick a random substring from corpus"""

  seqlen = cfg.getint('args', 'seqlen')
  seed_start = random.randint(0, len(text) - seqlen - 1)
  seed = text[seed_start: seed_start + seqlen]

  return seed

def sample_init_seq(model, init_char, chars, char2int, temp=0.01):
  """Sample a sequence of fixed length given a character"""

  seqlen = cfg.getint('args', 'seqlen')

  text = init_char
  for i in range(seqlen - len(text)):
    vectorized = np.zeros((1, len(text), len(chars)))
    for t, char in enumerate(text):
      vectorized[0, t, char2int[char]] = 1.

    preds = model.predict(vectorized)[0]
    next_index = sample_char(preds, temp)
    next_char = chars[next_index]
    text = text + next_char

  return text

def sample_seq(model,
               seed,
               chars,
               char2int,
               temp):
  """Generate n new characters from the model"""

  seqlen = cfg.getint('args', 'seqlen')

  sys.stdout.write('\nt = %f: ' % temp)
  # sys.stdout.write(seed)

  text = seed
  for i in range(cfg.getint('args', 'samples')):
    vectorized = np.zeros((1, seqlen, len(chars)))
    for t, char in enumerate(text):
      vectorized[0, t, char2int[char]] = 1.

    # feed it into the model
    preds = model.predict(vectorized)[0]

    # determine what character got predicted
    next_index = sample_char(preds, temp)
    next_char = chars[next_index]

    # add new character to the text
    text = text + next_char
    text = text[1:]

    sys.stdout.write(next_char)
    sys.stdout.flush()

  print()

def train_and_sample(model, x, y, text, chars, char2int):
  """Train and generate now"""

  seqlen = cfg.getint('args', 'seqlen') # chars in a sequence
  epochs = cfg.getint('args', 'epochs') # epochs to train

  # training loop; sample after each epoch
  for epoch in range(1, epochs):
    model.fit(x, y, batch_size=128, epochs=1, verbose=0)

    if cfg.get('args', 'initseq') == 'substring':
      seed = pick_init_seq(text)
    else:
      seed = sample_init_seq(model, ' ', chars, char2int)
    print('\nepoch: %d - seed: \'%s\'' % (epoch, seed))

    for temp in [0.1, 0.5, 1.0]:
        sample_seq(model, seed, chars, char2int, temp)

def main():
  """Driver function"""

  text = get_corpus()
  num_features = len(set(text))
  uniq_chars = sorted(list(set(text)))
  char2int = make_char_alphabet(text)

  x, y = make_training_data(text, char2int)
  model = get_model(num_features)
  train_and_sample(model, x, y, text, uniq_chars, char2int)

if __name__ == "__main__":

  # global settings from config file
  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])

  # driver function
  main()
