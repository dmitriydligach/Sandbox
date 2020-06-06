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
import os, configparser, math, random

from sklearn.metrics import accuracy_score

import imdb, utils

# deterministic determinism
torch.manual_seed(2020)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(2020)
random.seed(2020)

class TransformerClassifier(nn.Module):
  """A transformative experience"""

  def __init__(self, num_classes=2):
    """We have some of the best constructors in the world"""

    super(TransformerClassifier, self).__init__()

    tok = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab_size = tok.vocab_size # the reason we need bert tokenizer

    self.embedding = nn.Embedding(
      num_embeddings=vocab_size,
      embedding_dim=cfg.getint('model', 'emb_dim'))

    self.position = PositionalEncoding(
      embedding_dim=cfg.getint('model', 'emb_dim'))

    encoder_layer = nn.TransformerEncoderLayer(
      d_model=cfg.getint('model', 'emb_dim'),
      nhead=cfg.getint('model', 'num_heads'),
      dim_feedforward=cfg.getint('model', 'feedforw_dim'))

    self.trans_encoder = nn.TransformerEncoder(
      encoder_layer=encoder_layer,
      num_layers=cfg.getint('model', 'num_layers'))

    self.dropout = nn.Dropout(cfg.getfloat('model', 'dropout'))

    self.linear = nn.Linear(
      in_features=cfg.getint('model', 'emb_dim'),
      out_features=num_classes)

  def forward(self, texts, attention_mask):
    """Moving forward"""

    sqrtn = math.sqrt(cfg.getint('model', 'emb_dim'))
    output = self.embedding(texts) * sqrtn
    output = self.position(output)

    # encoder input: (seq_len, batch_size, emb_dim)
    # encoder output: (seq_len, batch_size, emb_dim)
    output = output.permute(1, 0, 2)
    output = self.trans_encoder(output, attention_mask)

    # average pooling
    # output = torch.mean(output, dim=0)

    # extract CLS token only
    output = output[0, :, :]

    output = self.dropout(output)
    output = self.linear(output)

    return output

class PositionalEncoding(nn.Module):
  """That's my position"""

  def __init__(self, embedding_dim):
    """Deconstructing the construct"""

    super(PositionalEncoding, self).__init__()

    self.dropout = nn.Dropout(p=0.1)

    max_len = 512 # bert tokenizer can't handle more
    pe = torch.zeros(max_len, embedding_dim)

    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * \
                         (-math.log(10000.0) / embedding_dim))

    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)

    self.register_buffer('pe', pe)

  def forward(self, x):
    """We're being very forward here"""

    x = x + self.pe[:x.size(0), :]

    return self.dropout(x)

def make_data_loader(texts, labels, sampler):
  """DataLoader objects for train or dev/test sets"""

  input_ids, attention_mask = utils.to_transformer_inputs(texts)
  labels = torch.tensor(labels)

  tensor_dataset = TensorDataset(input_ids, attention_mask, labels)
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
      batch_inputs, batch_mask, batch_labels = batch
      batch_mask = batch_mask.repeat(cfg.getint('model', 'num_heads'), 1, 1)
      optimizer.zero_grad()

      logits = model(batch_inputs, batch_mask)
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
    batch_inputs, batch_mask, batch_labels = batch
    batch_mask = batch_mask.repeat(cfg.getint('model', 'num_heads'), 1, 1)

    with torch.no_grad():
      logits = model(batch_inputs, batch_mask)
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

  model = TransformerClassifier()

  label_counts = torch.bincount(torch.IntTensor(tr_labels))
  weights = len(tr_labels) / (2.0 * label_counts)

  train(model, train_loader, val_loader, weights)

if __name__ == "__main__":

  cfg = configparser.ConfigParser()
  cfg.read(sys.argv[1])
  base = os.environ['DATA_ROOT']

  main()
