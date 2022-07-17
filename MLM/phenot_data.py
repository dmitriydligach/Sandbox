#!/usr/bin/env python3

import os, pathlib
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class PhenotypingDataset(Dataset):
  """Read data from files and make inputs/outputs"""

  def __init__(self, corpus_path, tokenizer_path):
    """Load tokenizer and save corpus path"""

    self.x = []
    self.y = []

    self.label2int = {'no':0, 'yes':1}
    self.corpus_path = corpus_path
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
    for d in os.listdir(self.corpus_path):
      label_dir = os.path.join(self.corpus_path, d)

      for f in os.listdir(label_dir):
        int_label = self.label2int[d.lower()]
        self.y.append(int_label)

        file_path = os.path.join(label_dir, f)
        text = pathlib.Path(file_path).read_text()

        cui_list = text.split()
        cui_counts.append(len(cui_list))
        cui_list = [cui[1:] for cui in cui_list[:510]]
        self.x.append(' '.join(cui_list))

    print('average number of cuis:', sum(cui_counts) / len(cui_counts))

if __name__ == "__main__":

  base = os.environ['DATA_ROOT']
  data_dir = os.path.join(base, 'Opioids1k/Train/')
  tokenizer_path = 'Tokenizer'

  dp = PhenotypingDataset(data_dir, tokenizer_path)
  print(dp[111])
