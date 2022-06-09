#!/usr/bin/env python3

import os
from pathlib import Path

from transformers import BertTokenizer
from transformers import AutoTokenizer

from tokenizers import Tokenizer
from tokenizers import normalizers
from tokenizers.models import WordPiece
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordPieceTrainer

def train_from_bert_tokenizer():
  """Source: https://huggingface.co/course/chapter6/2?fw=pt"""

  base = os.environ['DATA_ROOT']
  corpus_path = base + 'MimicIII/Encounters/Text/'

  orig_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

  files = (str(file) for file in Path(corpus_path).glob('*.txt'))
  new_tokenizer = orig_tokenizer.train_new_from_iterator(files, 30522)
  new_tokenizer.save_pretrained('Tokenizer')

def train():
  """Source: https://huggingface.co/docs/tokenizers/pipeline"""

  base = os.environ['DATA_ROOT']
  corpus_path = base + 'MimicIII/Encounters/Text/'

  bert_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))

  # input to tokenizer.encode() goes through this pipeline:
  # normalization, pre-tokenization, model, post-processing
  bert_tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
  bert_tokenizer.pre_tokenizer = Whitespace()
  bert_tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[("[CLS]", 1), ("[SEP]", 2)])

  files = [str(file) for file in Path(corpus_path).glob('*.txt')]
  trainer = WordPieceTrainer(
    vocab_size=30522,
    show_progress=True,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
  )
  bert_tokenizer.train(files, trainer)

  os.mkdir('./Tokenizer')
  bert_tokenizer.save("Tokenizer/tokenizer.json")

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

  train_from_bert_tokenizer()
  # train()
  # test()
