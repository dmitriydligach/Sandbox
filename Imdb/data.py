#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True

import os, configparser, glob

label2int = {'neg':0, 'pos':1}
int2label = {0:'neg', 1:'pos'}

class ImdbData:
  """Make x and y"""

  def __init__(
    self,
    dir_path,
    partition,
    n_files='all'):
    """"Deconstructing unconstructable"""

    self.pos_dir = os.path.join(dir_path, partition, 'pos/')
    self.neg_dir = os.path.join(dir_path, partition, 'neg/')
    self.n_files = None if n_files == 'all' else int(n_files)

  def read(self):
    """Make x, y"""

    texts = []
    labels = []

    for file in glob.glob(self.pos_dir + '*.txt')[:self.n_files]:
      texts.append(open(file).read().rstrip())
      labels.append(1)
    for file in glob.glob(self.neg_dir + '*.txt')[:self.n_files]:
      texts.append(open(file).read().rstrip())
      labels.append(0)

    return texts, labels

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  dat = ImdbData(
    os.path.join(base, cfg.get('data', 'dir_path')),
    partition='train',
    n_files=cfg.get('data', 'n_files'))

  texts, labels = dat.read()

  print('texts:\n', texts[-4:])
  print('labels:\n', labels[-4:])

  import collections
  print('unique labels:', collections.Counter(labels))

  max_len = max(len(text.split()) for text in texts)
  print('longest sequence:', max_len)

  mean_len = sum(len(text.split()) for text in texts) / len(texts)
  print('mean sequence:', mean_len)
