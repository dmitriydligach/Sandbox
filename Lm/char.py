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

# Length of extracted character sequences
maxlen = 60
# We sample a new sequence every `step` characters
step = 3

# ignore sklearn warnings
def warn(*args, **kwargs):
  pass
import warnings
warnings.warn = warn

def get_model(chars):
  """Model definition"""

  model = keras.models.Sequential()
  model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
  model.add(layers.Dense(len(chars), activation='softmax'))
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
      print('epoch', epoch)
      # Fit the model for 1 epoch on the available training data
      model.fit(x, y,
                batch_size=128,
                epochs=1)

      # Select a text seed at random
      start_index = random.randint(0, len(text) - maxlen - 1)
      generated_text = text[start_index: start_index + maxlen]
      print('--- Generating with seed: "' + generated_text + '"')

      for temperature in [0.2, 0.5, 1.0, 1.2]:
          print('------ temperature:', temperature)
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

def get_corpus():
  """Raw corpus"""

  path = keras.utils.get_file(
      'nietzsche.txt',
      origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
  return open(path).read().lower()

def get_training_data(text):
  """Make x and y"""

  # This holds our extracted sequences
  sentences = []
  # This holds the targets (the follow-up characters)
  next_chars = []

  for i in range(0, len(text) - maxlen, step):
      sentences.append(text[i: i + maxlen])
      next_chars.append(text[i + maxlen])
  print('Number of sequences:', len(sentences))

  # List of unique characters in the corpus
  chars = sorted(list(set(text)))
  print('Unique characters:', len(chars))
  # Dictionary mapping unique characters to their index in `chars`
  char_indices = dict((char, chars.index(char)) for char in chars)

  # Next, one-hot encode the characters into binary arrays.
  print('Vectorization...')
  x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
  y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
  for i, sentence in enumerate(sentences):
      for t, char in enumerate(sentence):
          x[i, t, char_indices[char]] = 1
      y[i, char_indices[next_chars[i]]] = 1

  return x, y, char_indices

if __name__ == "__main__":

  text = get_corpus()
  x, y, char_indices = get_training_data(text)
  model = get_model(sorted(list(set(text))))
  train_and_generate(model, x, y, sorted(list(set(text))), char_indices)
