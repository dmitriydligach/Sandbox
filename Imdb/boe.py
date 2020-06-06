#!/usr/bin/env python3

import sys
sys.dont_write_bytecode = True
sys.path.append('../Lib/')

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import RandomSampler, SequentialSampler

from transformers import BertTokenizer

import numpy as np
import os, configparser, random

from sklearn.metrics import accuracy_score

import imdb, utils

# deterministic determinism
torch.manual_seed(2020)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(2020)
random.seed(2020)

class BagOfEmbeddings(nn.Module):

  def __init__(self, num_class=2):
    """Constructor"""

    super(BagOfEmbeddings, self).__init__()

    tok = BertTokenizer.from_pretrained('bert-base-uncased')
    self.embed = nn.Embedding(
      num_embeddings=tok.vocab_size,
      embedding_dim=cfg.getint('model', 'emb_dim'))

    self.hidden1 = nn.Linear(
      in_features=cfg.getint('model', 'emb_dim'),
      out_features=cfg.getint('model', 'hidden_size'))

    self.hidden2 = nn.Linear(
      in_features=cfg.getint('model', 'hidden_size'),
      out_features=cfg.getint('model', 'hidden_size'))

    self.dropout = torch.nn.Dropout(cfg.getfloat('model', 'dropout'))

    self.classif = nn.Linear(
      in_features=cfg.getint('model', 'hidden_size'),
      out_features=num_class)

  def forward(self, texts):
    """Forward pass"""

    output = self.embed(texts)
    output = torch.mean(output, dim=1)
    output = self.hidden1(output)
    # output = self.hidden2(output)
    output = self.dropout(output)
    output = self.classif(output)

    return output

def make_data_loader(texts, labels, sampler):
  """DataLoader objects for train or dev/test sets"""

  input_ids = utils.to_token_id_sequences(texts)
  labels = torch.tensor(labels)

  tensor_dataset = TensorDataset(input_ids, labels)
  rnd_or_seq_sampler = sampler(tensor_dataset)

  data_loader = DataLoader(
    dataset=tensor_dataset,
    sampler=rnd_or_seq_sampler,
    batch_size=cfg.getint('model', 'batch_size'))

  return data_loader

def train(model, train_loader, val_loader, weights):
  """Training routine"""

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)

  weights = weights.to(device)
  cross_entropy_loss = torch.nn.CrossEntropyLoss(weights)

  optimizer = torch.optim.Adam(
    model.parameters(),
    lr=cfg.getfloat('model', 'lr'))

  for epoch in range(1, cfg.getint('model', 'num_epochs') + 1):
    model.train()
    train_loss, num_train_steps = 0, 0

    for batch in train_loader:
      batch = tuple(t.to(device) for t in batch)
      batch_inputs, batch_labels = batch
      optimizer.zero_grad()

      logits = model(batch_inputs)
      loss = cross_entropy_loss(logits, batch_labels)
      loss.backward()

      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()

      train_loss += loss.item()
      num_train_steps += 1

    av_loss = train_loss / num_train_steps
    val_loss, acc = evaluate(model, val_loader, weights)
    print('ep: %d, steps: %d, tr loss: %.3f, val loss: %.3f, val acc: %.3f' % \
          (epoch, num_train_steps, av_loss, val_loss, acc))

def evaluate(model, data_loader, weights, suppress_output=True):
  """Evaluation routine"""

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  weights = weights.to(device)
  model.to(device)

  cross_entropy_loss = torch.nn.CrossEntropyLoss(weights)
  total_loss, num_steps = 0, 0

  model.eval()

  all_labels = []
  all_predictions = []

  for batch in data_loader:
    batch = tuple(t.to(device) for t in batch)
    batch_inputs, batch_labels = batch

    with torch.no_grad():
      logits = model(batch_inputs)
      loss = cross_entropy_loss(logits, batch_labels)

    batch_logits = logits.detach().cpu().numpy()
    batch_labels = batch_labels.to('cpu').numpy()
    batch_preds = np.argmax(batch_logits, axis=1)

    all_labels.extend(batch_labels.tolist())
    all_predictions.extend(batch_preds.tolist())

    total_loss += loss.item()
    num_steps += 1

  accuracy = accuracy_score(all_labels, all_predictions)
  return total_loss / num_steps, accuracy

def main():
  """Fine-tune bert"""

  train_data = imdb.ImdbData(
    os.path.join(base, cfg.get('data', 'dir_path')),
    partition='train',
    n_files=cfg.get('data', 'n_files'))
  tr_texts, tr_labels = train_data.read()
  train_loader = make_data_loader(tr_texts, tr_labels, RandomSampler)

  val_data = imdb.ImdbData(
    os.path.join(base, cfg.get('data', 'dir_path')),
    partition='test',
    n_files=cfg.get('data', 'n_files'))
  val_texts, val_labels = val_data.read()
  val_loader = make_data_loader(val_texts, val_labels, SequentialSampler)

  print('loaded %d training and %d validation samples' % \
        (len(tr_texts), len(val_texts)))

  model = BagOfEmbeddings()

  label_counts = torch.bincount(torch.IntTensor(tr_labels))
  weights = len(tr_labels) / (2.0 * label_counts)

  train(model, train_loader, val_loader, weights)

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  main()
