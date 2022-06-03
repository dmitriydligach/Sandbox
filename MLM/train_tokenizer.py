#!/usr/bin/env python3

import os
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from transformers import BertTokenizer

# Todo: update for BERT
# Todo: tie to "tokenizer_class": "BertTokenizer"

def train():
  """Using the latest tokenizers API"""

  base = os.environ['DATA_ROOT']
  corpus_path = base + 'MimicIII/Encounters/Text/'
  files = [str(file) for file in Path(corpus_path).glob('*.txt')]

  tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

  trainer = BpeTrainer(
    vocab_size=30000,
    min_frequency=2,
    show_progress=True,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

  tokenizer.pre_tokenizer = Whitespace()

  tokenizer.train(files, trainer)

  os.mkdir('./Tokenizer')
  tokenizer.save("Tokenizer/tokenizer.json")

def test():
  """Using the new API"""

  # load as a generic tokenizer
  tokenizer = Tokenizer.from_file("Tokenizer/tokenizer.json")
  output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
  print(output.tokens)
  print(output.ids)

  # load as a bert tokenizer
  tokenizer = BertTokenizer.from_pretrained('Tokenizer/tokenizer.json')
  output = tokenizer.encode('Patient complains about being nervous')
  print(output)

if __name__ == "__main__":

  # train()
  test()
