#!/usr/bin/env python3

import os, pathlib, numpy
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# maximum sequence length
max_length = 512

class PhenotypingDataset(Dataset):
  """Read data from files and make inputs/outputs"""

  def __init__(self, corpus_path_or_files, tokenizer_path, print_stats=False):
    """Load tokenizer and save corpus path"""

    self.x = []
    self.y = []

    self.file_paths = []

    if type(corpus_path_or_files) == str:
      self.file_paths = pathlib.Path(corpus_path_or_files).glob('*/*.txt')
    elif type(corpus_path_or_files) == list:
      self.file_paths = corpus_path_or_files
    else:
      raise ValueError('Invalid corpus path!')

    self.label2int = {'no':0, 'yes':1}
    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    self.load_examples(print_stats=print_stats)

  def __len__(self):
    """Requried by pytorch"""

    assert(len(self.x) == len(self.y))
    return len(self.x)

  def __getitem__(self, index):
    """Required by pytorch"""

    output = self.tokenizer(
      self.x[index],
      max_length=max_length,
      padding='max_length',
      truncation=True,
      return_tensors='pt')

    return dict(
      input_ids = output.input_ids.squeeze(),
      attention_mask = output.attention_mask.squeeze(),
      labels = self.y[index])

  def load_examples(self, print_stats=False):
    """Convert examples to lists of indices and get token count stats"""

    cui_counts = [] # n cuis in each sample

    for file_path in self.file_paths:
      str_label = file_path.parts[-2].lower()
      int_label = self.label2int[str_label]
      self.y.append(int_label)

      text = file_path.read_text()
      cui_list = text.split()
      cui_counts.append(len(cui_list))

      # strip 'C' and get the first max_length-2 (e.g. 510) CUIs
      cui_list = [cui[1:] for cui in cui_list[:max_length-2]]
      self.x.append(' '.join(cui_list))

    if print_stats:
      print('num of training examples:', len(cui_counts))
      print('mean num of cuis:', numpy.mean(cui_counts))
      print('median num of cuis:', numpy.median(cui_counts))
      print('min num of cuis:', numpy.min(cui_counts))
      print('max num of cuis:', numpy.max(cui_counts))
      print('standard deviation:', numpy.std(cui_counts))

def analyze_datasets():
  """Print stats about phenotyping datasets"""

  base = os.environ['DATA_ROOT']
  tokenizer_path = 'CuiTokenizer'

  eval_datasets = [('alcohol', 'Alcohol/anc_notes_cuis/', 'Alcohol/anc_notes_test_cuis/'),
                   ('ards', 'Ards/Train/', 'Ards/Test/'),
                   ('injury', 'Injury/Train/', 'Injury/Test/'),
                   ('opioids', 'Opioids1k/Train/', 'Opioids1k/Test/')]

  for name, train, test in eval_datasets:
    print('\n##### %s #####' % name)
    data_dir = os.path.join(base, train)
    dp = PhenotypingDataset(data_dir, tokenizer_path, print_stats=True)

if __name__ == "__main__":

  analyze_datasets()