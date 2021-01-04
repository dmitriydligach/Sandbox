#!/usr/bin/env python3

import sys
sys.path.append('../Lib/')

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

import random, argparse
import wikidata

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup)

# deterministic determinism
torch.manual_seed(2020)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
random.seed(2020)

class T5FineTuner(nn.Module):
  """A transformative experience"""

  def __init__(self):
    """Some of the best constructors in the world"""

    super(T5FineTuner, self).__init__()
    self.model = T5ForConditionalGeneration.from_pretrained(
      hparams.model_name)

  def forward(
   self,
   input_ids,
   attention_mask,
   decoder_input_ids,
   decoder_attention_mask,
   labels):
    """Forwarding"""

    output = self.model(
      input_ids=input_ids,
      attention_mask=attention_mask,
      decoder_input_ids=decoder_input_ids,
      decoder_attention_mask=decoder_attention_mask,
      labels=labels)

    return output

def fit(model, train_loader, val_loader, tokenizer):
  """Training routine"""

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)

  optimizer = AdamW(model.parameters())

  for epoch in range(1, hparams.n_epochs + 1):
    train_loss, num_train_steps = 0, 0
    model.train()

    for batch in train_loader:
      optimizer.zero_grad()

      batch = tuple(t.to(device) for t in batch)
      source_ids, source_mask, target_ids, target_mask = batch

      labels = target_ids
      labels[labels[:, :] == tokenizer.pad_token_id] = -100

      outputs = model(
        input_ids=source_ids,
        attention_mask=source_mask,
        decoder_input_ids=None,
        decoder_attention_mask=target_mask,
        labels=labels)
      loss = outputs[0]

      loss.backward()

      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimizer.step()

      train_loss += loss.item()
      num_train_steps += 1

    av_loss = train_loss / num_train_steps
    val_loss = evaluate(model, val_loader, tokenizer)
    print('ep: %d, steps: %d, tr loss: %.3f, val loss: %.3f' % \
          (epoch, num_train_steps, av_loss, val_loss))

def evaluate(model, data_loader, tokenizer):
  """Evaluation routine"""

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)

  total_loss, num_steps = 0, 0
  model.eval()

  for batch in data_loader:
    batch = tuple(t.to(device) for t in batch)
    source_ids, source_mask, target_ids, target_mask = batch

    labels = target_ids
    labels[labels[:, :] == tokenizer.pad_token_id] = -100

    with torch.no_grad():
      outputs = model(
        input_ids=source_ids,
        attention_mask=source_mask,
        decoder_input_ids=None,
        decoder_attention_mask=target_mask,
        labels=labels)
      loss = outputs[0]

    total_loss += loss.item()
    num_steps += 1

  average_loss = total_loss / num_steps
  return average_loss

def generate(model, data_loader, tokenizer):
  """Need to add 'summarize' if run before training"""

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  model.eval()

  for batch in data_loader:
    batch = tuple(t.to(device) for t in batch)
    source_ids, source_mask, target_ids, target_mask = batch

    inputs = tokenizer.batch_decode(source_ids)
    targets = tokenizer.batch_decode(target_ids)

    predictions = model.model.generate(
      input_ids=source_ids,
      max_length=hparams.max_output_length,
      early_stopping=True,
      num_beams=2,
      attention_mask=source_mask)
    predictions = tokenizer.batch_decode(
      predictions,
      skip_special_tokens=True,
      clean_up_tokenization_spaces=True)

    print('input:', inputs[0])
    print('target:', targets[0])
    print('prediction:', predictions[0])
    print()

def main():
  """Fine-tune on summarization data"""

  tokenizer = T5Tokenizer.from_pretrained(hparams.model_name)

  train_dataset = wikidata.WikiHow(tokenizer=tokenizer, split='train')
  train_data_loader = DataLoader(
    train_dataset,
    batch_size=hparams.batch_size)

  val_dataset = wikidata.WikiHow(tokenizer=tokenizer, split='validation')
  val_data_loader = DataLoader(
    val_dataset,
    batch_size=hparams.batch_size)

  model = T5FineTuner()

  fit(model, train_data_loader, val_data_loader, tokenizer)
  generate(model, val_data_loader, tokenizer)

if __name__ == "__main__":
  "My kind of street"

  hparam_dict = dict(
    model_name='t5-base',
    max_input_length=512,
    max_output_length=100,
    batch_size=32,
    n_epochs=5)
  hparams = argparse.Namespace(**hparam_dict)
  print('hyper-parameters:', hparams)

  main()
