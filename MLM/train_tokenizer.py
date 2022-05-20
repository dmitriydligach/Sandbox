#!/usr/bin/env python3

import os
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

def train():
  """My main man"""

  base = os.environ['DATA_ROOT']
  corpus_path = base + 'MimicIII/Encounters/Text/'
  files = [str(file) for file in Path(corpus_path).glob('*.txt')]

  tokenizer = ByteLevelBPETokenizer(lowercase=True)
  tokenizer.train(
    files=files,
    vocab_size=30000,
    min_frequency=2,
    show_progress=True,
    special_tokens=[
      "<s>",
      "</s>",
      "<pad>",
      "<unk>",
      "<mask>"])

  os.mkdir('./Tokenizer')
  tokenizer.save_model('./Tokenizer')

def test():
  """Test trained tokenizer"""

  tokenizer = ByteLevelBPETokenizer(
    './Tokenizer/vocab.json',
    './Tokenizer/merges.txt')

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
