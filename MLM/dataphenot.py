#!/usr/bin/env python3

import os, pathlib
from transformers import AutoTokenizer

class DatasetProvider:
  """Read data from files and make inputs/outputs"""

  def __init__(self,
               corpus_path,
               tokenizer_path):
    """Load tokenizer and save corpus path"""

    self.corpus_path = corpus_path
    self.label2int = {'no':0, 'yes':1}
    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

  def load_as_int_seqs(self):
    """Convert examples into lists of indices"""

    x = []
    y = []

    cui_counts = [] # n cuis in each sample

    for d in os.listdir(self.corpus_path):
      label_dir = os.path.join(self.corpus_path, d)

      for f in os.listdir(label_dir):
        int_label = self.label2int[d.lower()]
        y.append(int_label)

        file_path = os.path.join(label_dir, f)
        text = pathlib.Path(file_path).read_text()

        cui_list = text.split()
        cui_counts.append(len(cui_list))
        cui_list = [cui[1:] for cui in cui_list[:510]]

        first_510_cuis_string = ' '.join(cui_list)
        x.append(self.tokenizer.encode(first_510_cuis_string))

    print('average number of cuis:', sum(cui_counts) / len(cui_counts))

    return x, y

if __name__ == "__main__":

  base = os.environ['DATA_ROOT']
  data_dir = os.path.join(base, 'Opioids1k/Train/')

  dp = DatasetProvider(data_dir, 'Tokenizer')
  x, y = dp.load_as_int_seqs()
