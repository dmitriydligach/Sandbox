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
  return open(path).read().lower().replace('\n', ' ')

def make_char_alphabet(text):
  """Map characters to integers"""

  chars = sorted(list(set(text)))
  return dict((char, chars.index(char)) for char in chars)

def make_training_data(text, char2int):
  """Make x and y"""

  maxlen = cfg.getint('args', 'maxlen') # chars in a sequence

  sequences = [] # sequences of maxlen characters
  targets = []   # char the follows each sequence above

  step = cfg.getint('args', 'step')
  for i in range(0, len(text) - maxlen, step):
      sequences.append(text[i: i + maxlen])
      targets.append(text[i + maxlen])

  # vectorize sequences; make a tensor of the following shape:
  # (samples, time_steps, features) -> (samples, maxlen, uniq_chars))
  items = len(sequences) * maxlen * len(char2int)
  item_size_in_bytes = np.dtype(np.bool).itemsize
  print('train tensor shape:', (len(sequences), maxlen, len(char2int)))
  print('allocating:', hurry.filesize.size(items * item_size_in_bytes))
  x = np.zeros((len(sequences), maxlen, len(char2int)), dtype=np.bool)
  print('train tensor size in bytes:', hurry.filesize.size(x.nbytes))
  y = np.zeros((len(sequences), len(char2int)), dtype=np.bool)

  for n, sequence in enumerate(sequences):
      for time_step, char in enumerate(sequence):
          x[n, time_step, char2int[char]] = 1
      y[n, char2int[targets[n]]] = 1

  return x, y

def get_model(num_features):
  """Model that takes (time_steps, features) as input"""

  maxlen = cfg.getint('args', 'maxlen') # chars in a sequence

  model = keras.models.Sequential()
  model.add(layers.LSTM(128, input_shape=(maxlen, num_features)))
  model.add(layers.Dense(num_features, activation='softmax'))
  optimizer = keras.optimizers.RMSprop(lr=0.01)
  model.compile(loss='categorical_crossentropy', optimizer=optimizer)

  return model

def sample(preds, temperature=1.0):
  """Reweight the prob distribution and sample a char"""

  # preds shape: (len(chars),)
  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)

  return np.argmax(probas)

def generate_samples(model,
                     seed,
                     temperature,
                     chars,
                     char2int):
  """Generate n new characters from the model"""

  maxlen = cfg.getint('args', 'maxlen') # chars in a sequence

  sys.stdout.write('\nt = %f: ' % temperature)
  sys.stdout.write(seed)

  generated_text = seed
  samples = cfg.getint('args', 'samples')

  for i in range(samples):
    # vectorize what we have so far
    sampled = np.zeros((1, maxlen, len(chars)))
    for t, char in enumerate(generated_text):
        sampled[0, t, char2int[char]] = 1.

    # feed it into the model
    preds = model.predict(sampled)[0]

    # determine what character got predicted
    next_index = sample(preds, temperature)
    next_char = chars[next_index]

    # add new character to the text
    generated_text = generated_text + next_char
    generated_text = generated_text[1:]

    sys.stdout.write(next_char)
    sys.stdout.flush()

  print()

def train_and_generate(model, x, y, chars, char2int):
  """Train and generate now"""

  maxlen = cfg.getint('args', 'maxlen') # chars in a sequence
  epochs = cfg.getint('args', 'step')   # epochs to train

  for epoch in range(1, epochs):
    model.fit(x, y, batch_size=128, epochs=1)
    seed = text[0]

    for temperature in [0.2, 0.5, 1.0, 1.2]:
        generate_samples(
          model,
          seed,
          temperature,
          chars,
          char2int)

if __name__ == "__main__":

  # global settings from config file
  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])

  text = get_corpus()
  num_features = len(set(text))
  uniq_chars = sorted(list(set(text)))
  char2int = make_char_alphabet(text)

  x, y = make_training_data(text, char2int)
  model = get_model(num_features)
  train_and_generate(model, x, y, uniq_chars, char2int)
