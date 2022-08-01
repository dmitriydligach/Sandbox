#!/usr/bin/env python3

import torch, random, phenot_data, os, shutil, pathlib, metrics
import numpy as np
from torch.nn import functional as F
from datasets import load_metric
from transformers import (TrainingArguments,
                          Trainer,
                          AutoModelForSequenceClassification,
                          IntervalStrategy)

# deterministic determinism
torch.manual_seed(2022)
random.seed(2022)

# misc constants
pretrained_model_path = 'PreTrainedModel/'
fine_tuned_model_path = 'FineTunedModel/'

# hyperparameters
n_epochs = 15
batch_size = 48
learning_rate = 5e-5

def compute_metrics(eval_pred):
  """Compute custom evaluation metric"""

  logits, labels = eval_pred
  logits = torch.from_numpy(logits)
  probabilities = F.softmax(logits, dim=1).numpy()[:, 1]

  metric = load_metric('roc_auc')
  return metric.compute(prediction_scores=probabilities, references=labels)

def train_test_split(dir_path, train_size=0.8):
  """Split files in a directory into train/test"""

  all_files = list(pathlib.Path(dir_path).glob('*/*.txt'))
  n_train_files = int(len(all_files) *  train_size)

  random.shuffle(all_files)
  train_files = all_files[:n_train_files]
  test_files = all_files[n_train_files:]

  return train_files, test_files

def init_transformer(m: torch.nn.Module):
  """Jiacheng Zhang's transformer initialization wisdom"""

  for name, params in m.named_parameters():
    print('initializing:', name)

    if len(params.shape) >= 2:
      torch.nn.init.xavier_uniform_(params)
    else:
      if 'bias' in name:
        torch.nn.init.zeros_(params)
      else:
        torch.nn.init.uniform_(params)

def model_selection():
  """Fine-tune on phenotyping data"""

  model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_path,
    num_labels=2)

  train_files, dev_files = train_test_split(train_dir)
  train_dataset = phenot_data.PhenotypingDataset(train_files, tok_path)
  dev_dataset = phenot_data.PhenotypingDataset(dev_files, tok_path)

  training_args = TrainingArguments(
    output_dir='./Results',
    overwrite_output_dir=True,
    num_train_epochs=n_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=learning_rate,
    load_best_model_at_end=True,
    save_strategy=IntervalStrategy.EPOCH,
    evaluation_strategy=IntervalStrategy.EPOCH,
    disable_tqdm=True)
  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics)
  trainer.train()

  print('\n*** Validation Loss ***\n')
  for entry in trainer.state.log_history:
    if 'eval_loss' in entry:
      print('epoch: %s, val loss: %s' % (entry['epoch'], entry['eval_loss']))
  print('best metric:', trainer.state.best_metric)

  print('\n*** Validation ROC AUC ***\n')
  for entry in trainer.state.log_history:
    if 'eval_roc_auc' in entry:
      print('epoch: %s, val loss: %s' % (entry['epoch'], entry['eval_roc_auc']))
  print('best metric:', trainer.state.best_metric)

  predictions = trainer.predict(dev_dataset)
  logits = torch.from_numpy(predictions.predictions)
  probabilities = F.softmax(logits, dim=1).numpy()[:, 1]
  labels = np.argmax(logits, axis=1)

  print('\n*** Evaluation results ***\n')
  metrics.report_accuracy(dev_dataset.y, labels)
  metrics.report_roc_auc(dev_dataset.y, probabilities)
  metrics.report_pr_auc(dev_dataset.y, probabilities)

def evaluation(best_n_epochs):
  """Fine-tune on phenotyping data"""

  # need this to save a fine-tuned model
  if os.path.isdir(fine_tuned_model_path):
    shutil.rmtree(fine_tuned_model_path)
  os.mkdir(fine_tuned_model_path)

  model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_path,
    num_labels=2)

  train_dataset = phenot_data.PhenotypingDataset(train_dir, tok_path)
  test_dataset = phenot_data.PhenotypingDataset(test_dir, tok_path)

  training_args = TrainingArguments(
    output_dir='./Results',
    overwrite_output_dir=True,
    num_train_epochs=best_n_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=learning_rate,
    load_best_model_at_end=True,
    save_strategy=IntervalStrategy.NO,
    evaluation_strategy=IntervalStrategy.NO,
    disable_tqdm=True)
  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset)
  trainer.train()
  trainer.save_model(fine_tuned_model_path)

  predictions = trainer.predict(test_dataset)
  logits = torch.from_numpy(predictions.predictions)
  probabilities = F.softmax(logits, dim=1).numpy()[:, 1]
  labels = np.argmax(logits, axis=1)

  print('\n*** Evaluation results ***\n')
  metrics.report_accuracy(test_dataset.y, labels)
  metrics.report_roc_auc(test_dataset.y, probabilities)
  metrics.report_pr_auc(test_dataset.y, probabilities)

if __name__ == "__main__":
  "My kind of street"

  base = os.environ['DATA_ROOT']
  train_dir = os.path.join(base, 'Opioids1k/Train/')
  test_dir = os.path.join(base, 'Opioids1k/Test/')
  tok_path = 'Tokenizer'

  model_selection()
  # evaluation(10)
