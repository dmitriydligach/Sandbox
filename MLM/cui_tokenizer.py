#!/usr/bin/env python3

import os, json, pathlib
from collections import Counter

old_tokenizer_path = 'OldTokenizer/tokenizer.json'
new_tokenizer_path = 'cui_tokenizer.json'

base = os.environ['DATA_ROOT']
mimic_cui_dir = os.path.join(base, 'MimicIII/Encounters/Cuis/All/')

vocab_size = 30522
n_special_tokens = 5

def make_cui_vocab():
  """Read CUI files and pick most frequent ones"""

  cui_counter = Counter()
  for cui_file in pathlib.Path(mimic_cui_dir).glob('*.txt'):
    text = pathlib.Path(cui_file).read_text()
    cui_counter.update(text.split())

  print('%d unique cuis found' % len(cui_counter))

  vocab = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4}
  cui_counts = cui_counter.most_common(vocab_size - n_special_tokens)
  for index, (cui, count) in enumerate(cui_counts):
    cui = cui[1:] # first character 'C' messes up the tokenizer
    vocab[cui] = index + n_special_tokens

  print('vocab size:', len(vocab))
  return vocab

def write_vocab_file():
  """Reading and writing"""

  # test tokenizer as follows:
  # tokenizer = AutoTokenizer.from_pretrained('Tokenizer')
  # tokenizer.encode('one two three')

  with open(old_tokenizer_path, 'r') as old_tokenizer_file:
    tokenizer_json = json.load(old_tokenizer_file)

  # print(tokenizer_json['model']['vocab']['##1'])
  # tokenizer_json['model']['vocab'] = \
  #   {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4,
  #    'C0231683': 5, 'C0302142': 6, 'C0040408': 7, 'C0445403': 8}

  tokenizer_json['model']['vocab'] = make_cui_vocab()

  with open(new_tokenizer_path, 'w') as new_tokenizer_file:
    json.dump(tokenizer_json, new_tokenizer_file, indent=2)

if __name__ == "__main__":

  write_vocab_file()
