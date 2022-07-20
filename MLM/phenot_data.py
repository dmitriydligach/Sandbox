#!/usr/bin/env python3

import os, pathlib
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class PhenotypingDataset(Dataset):
  """Read data from files and make inputs/outputs"""

  def __init__(self, corpus_path_or_files, tokenizer_path):
    """Load tokenizer and save corpus path"""

    self.x = []
    self.y = []

    self.file_paths = []

    if type(corpus_path_or_files) == str:
      self.file_paths = pathlib.Path(corpus_path_or_files).glob('*/*.txt')
    elif type(corpus_path_or_files) == list:
      self.file_paths = corpus_path_or_files
    else:
      print('wrong type!')

    self.label2int = {'no':0, 'yes':1}
    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    self.load_examples()

  def __len__(self):
    """Requried by pytorch"""

    assert(len(self.x) == len(self.y))
    return len(self.x)

  def __getitem__(self, index):
    """Required by pytorch"""

    output = self.tokenizer(
      self.x[index],
      max_length=512,
      padding='max_length',
      truncation=True,
      return_tensors='pt')

    return dict(
      input_ids = output.input_ids.squeeze(),
      attention_mask = output.attention_mask.squeeze(),
      labels = self.y[index])

  def load_examples(self):
    """Convert examples into lists of indices"""

    cui_counts = [] # n cuis in each sample

    for file_path in self.file_paths:
      str_label = file_path.parts[-2].lower()
      int_label = self.label2int[str_label]
      self.y.append(int_label)

      text = file_path.read_text()
      cui_list = text.split()
      cui_counts.append(len(cui_list))

      # strip 'C' and get the first 510 CUIs
      cui_list = [cui[1:] for cui in cui_list[:510]]
      self.x.append(' '.join(cui_list))

    print('average number of cuis:', sum(cui_counts) / len(cui_counts))

if __name__ == "__main__":

  base = os.environ['DATA_ROOT']
  data_dir = os.path.join(base, 'Opioids1k/Train/')
  tokenizer_path = 'Tokenizer'

  dp = PhenotypingDataset(data_dir, tokenizer_path)
  print(dp[111])
