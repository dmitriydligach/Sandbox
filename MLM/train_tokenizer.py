#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True
sys.path.append('../Anafora')

import glob, os

from tokenizers import CharBPETokenizer

def train():
  """My main man"""

  base = os.environ['DATA_ROOT']
  corpus_path = base + 'Thyme/Text/train+dev+test/*'
  files = glob.glob(corpus_path)

  tokenizer = CharBPETokenizer(lowercase=True)
  tokenizer.train(
    files=files,
    vocab_size=10000,
    min_frequency=3,
    show_progress=True)
  tokenizer.save('./thyme-tokenizer')


def test():
  """Test trained tokenizer"""

  tokenizer = CharBPETokenizer(
    './thyme-tokenizer-vocab.json',
    './thyme-tokenizer-merges.txt')

  vocab = tokenizer.get_vocab()
  print('vocab size:', len(vocab))

  encoded = tokenizer.encode('patient dr. who diagnosed with brain abc')
  encoded.pad(15)

  print('encoded:', encoded.ids)
  print('decoded:', tokenizer.decode(encoded.ids))

  print(encoded.tokens)
  print(encoded.attention_mask)

if __name__ == "__main__":

  train()
  test()
