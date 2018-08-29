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

# the rest of the imports
import keras, random, sys
from keras import layers

maxlen = 60 # char sequence length
step = 3    # sample new sequence very 3 chars

def get_corpus():
  """Raw corpus"""

  path = keras.utils.get_file(
      'nietzsche.txt',
      origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
  return open(path).read().lower()

def make_training_data(text):
  """Make x and y"""

  sequences = [] # sequences of maxlen characters
  targets = []   # char the follows each sequence above

  for i in range(0, len(text) - maxlen, step):
      sequences.append(text[i: i + maxlen])
      targets.append(text[i + maxlen])

  chars = sorted(list(set(text)))
  char2index = dict((char, chars.index(char)) for char in chars)

  # vectorize sequences; make a tensor of the following shape:
  # (samples, time_steps, features) -> (samples, maxlen, len(chars))
  x = np.zeros((len(sequences), maxlen, len(chars)), dtype=np.bool)
  y = np.zeros((len(sequences), len(chars)), dtype=np.bool)

  for n, sequence in enumerate(sequences):
      for time_step, char in enumerate(sequence):
          x[n, time_step, char2index[char]] = 1
      y[n, char2index[targets[n]]] = 1

  return char2index, x, y

def get_model(num_features):
  """Model that takes (time_steps, features) as input"""

  model = keras.models.Sequential()
  model.add(layers.LSTM(128, input_shape=(maxlen, num_features)))
  model.add(layers.Dense(num_features, activation='softmax'))
  optimizer = keras.optimizers.RMSprop(lr=0.01)
  model.compile(loss='categorical_crossentropy', optimizer=optimizer)

  return model

def sample(preds, temperature=1.0):
  """Reweight the prob distribution and sample a char"""

  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)

  return np.argmax(probas)

def train_and_generate(model, x, y, chars, char_indices):
  """Train and generate now"""

  for epoch in range(1, 60):

      model.fit(x, y, batch_size=128, epochs=1)

      start_index = random.randint(0, len(text) - maxlen - 1)
      generated_text = text[start_index: start_index + maxlen]
      print('seed:' + generated_text)

      for temperature in [0.2, 0.5, 1.0, 1.2]:
          print('temperature:', temperature)
          sys.stdout.write(generated_text)

          # We generate 400 characters
          for i in range(400):
              sampled = np.zeros((1, maxlen, len(chars)))
              for t, char in enumerate(generated_text):
                  sampled[0, t, char_indices[char]] = 1.

              preds = model.predict(sampled, verbose=0)[0]
              next_index = sample(preds, temperature)
              next_char = chars[next_index]

              generated_text += next_char
              generated_text = generated_text[1:]

              sys.stdout.write(next_char)
              sys.stdout.flush()
          print()

if __name__ == "__main__":

  text = get_corpus()
  char2index, x, y = make_training_data(text)
  model = get_model(len(set(text)))
  train_and_generate(model, x, y, sorted(list(set(text))), char2index)
