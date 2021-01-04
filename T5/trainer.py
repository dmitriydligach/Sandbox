#!/usr/bin/env python3

import argparse
import sys
import time

sys.path.append('../Lib/')

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

import random
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

model_name = 't5-small'

class T5FineTuner(nn.Module):
  """A transformative experience"""

  def __init__(self):
    """Some of the best constructors in the world"""

    super(T5FineTuner, self).__init__()
    self.model = T5ForConditionalGeneration.from_pretrained(model_name)

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

def fit(model, train_loader, val_loader, tokenizer, n_epochs):
  """Training routine"""

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)

  optimizer = AdamW(model.parameters())

  for epoch in range(1, n_epochs + 1):
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

  av_loss = total_loss / num_steps

  return av_loss

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
    print('input:', inputs[0])
    print('target:', targets[0])

    predictions = model.model.generate(
      input_ids=source_ids,
      max_length=50,
      early_stopping=True,
      num_beams=2,
      attention_mask=source_mask)
    predictions = tokenizer.batch_decode(
      predictions,
      skip_special_tokens=True,
      clean_up_tokenization_spaces=True)
    print('predictions:', predictions[0])
    print()

def run_it():
  """Fine-tune on summarization data"""

  tokenizer = T5Tokenizer.from_pretrained(model_name)

  train_dataset = wikidata.WikiHow(tokenizer=tokenizer, split='train')
  train_data_loader = DataLoader( # batch generator
    train_dataset,
    batch_size=64)

  val_dataset = wikidata.WikiHow(tokenizer=tokenizer, split='validation')
  val_data_loader = DataLoader( # batch generator
    val_dataset,
    batch_size=64)

  model = T5FineTuner()
  fit(model, train_data_loader, val_data_loader, tokenizer, n_epochs=3)
  generate(model, val_data_loader, tokenizer)

if __name__ == "__main__":
  "My kind of street"

  args_dict = dict(
    model_name_or_path='t5-small',
    tokenizer_name_or_path='t5-small',
    max_input_length=512,
    max_output_length=150,
    freeze_encoder=False,
    freeze_embeds=False,
    learning_rate=3e-4,
    weight_decay=0.0,
    adam_epsilon=1e-8,
    warmup_steps=0,
    train_batch_size=4,
    eval_batch_size=4,
    num_train_epochs=2,
    gradient_accumulation_steps=1,
    n_gpu=1,
    resume_from_checkpoint=None,
    val_check_interval=0.05,
    n_val=1000,
    n_train=-1,
    n_test=-1,
    early_stop_callback=False,
    fp_16=False,
    opt_level='O1',
    max_grad_norm=1.0,
    seed=42)
  args = argparse.Namespace(**args_dict)

  run_it()
