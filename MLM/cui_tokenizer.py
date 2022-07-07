#!/usr/bin/env python3

import json

old_tokenizer_path = 'OldTokenizer/tokenizer.json'
new_tokenizer_path = 'cui_tokenizer.json'

def main():
  """Reading and writing"""

  with open(old_tokenizer_path, 'r') as old_tokenizer_file:
    tokenizer_json = json.load(old_tokenizer_file)

  print(tokenizer_json['model']['vocab']['##1'])

  tokenizer_json['model']['vocab'] = \
    {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4,
     'one': 5, 'two': 6, 'three': 7, 'four': 8}

  with open(new_tokenizer_path, 'w') as new_tokenizer_file:
    json.dump(tokenizer_json, new_tokenizer_file, indent=2)

if __name__ == "__main__":

  main()