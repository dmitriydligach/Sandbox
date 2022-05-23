#!/usr/bin/env python3

import os
from pathlib import Path

# old tokenizers API?
from tokenizers import ByteLevelBPETokenizer

# new tokenizers API?
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def train():
  """Using the latest tokenizers API"""

  base = os.environ['DATA_ROOT']
  corpus_path = base + 'MimicIII/Encounters/Text/'
  files = [str(file) for file in Path(corpus_path).glob('*.txt')]

  tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
  trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
  tokenizer.pre_tokenizer = Whitespace()
  tokenizer.train(files, trainer)

  os.mkdir('./Tokenizer')
  tokenizer.save("Tokenizer/mimic.json")

def train_old():
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

def test_old():
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

def test():
  """Using the new API"""

  tokenizer = Tokenizer.from_file("Tokenizer/mimic.json")
  output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
  print(output.tokens)
  print(output.ids)

if __name__ == "__main__":

  # train()
  test()
