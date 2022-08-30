#!/usr/bin/env python3

import torch, random, phenot_data, os, shutil, pathlib, metrics
import numpy as np
from torch.nn import functional as F
from transformers import (TrainingArguments,
                          Trainer,
                          AutoModelForSequenceClassification,
                          IntervalStrategy)

# misc constants
pretrained_model_path = 'PreTrainedModel/'
metric_for_best_model = 'eval_pr_auc'
tokenizer_path = './CuiTokenizer'
results_file = './results.txt'

# hyperparameters
model_selection_n_epochs = 15
batch_size = 48
lr = 5e-5

# datasets (name, train path, test path) tuples
eval_datasets = [('alcohol', 'Alcohol/anc_notes_cuis/', 'Alcohol/anc_notes_test_cuis/'),
                 ('ards', 'Ards/Train/', 'Ards/Test/'),
                ('injury', 'Injury/Train/', 'Injury/Test/'),
                ('opioids', 'Opioids1k/Train/', 'Opioids1k/Test/')]

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

def train_test_split(dir_path, train_size=0.8):
  """Split files in a directory into train/test"""

  all_files = list(pathlib.Path(dir_path).glob('*/*.txt'))
  n_train_files = int(len(all_files) *  train_size)

  random.shuffle(all_files)
  train_files = all_files[:n_train_files]
  test_files = all_files[n_train_files:]

  return train_files, test_files

def compute_metrics(eval_pred):
  """Compute custom evaluation metric"""

  logits, labels = eval_pred
  logits = torch.from_numpy(logits) # torch tensor for softmax
  probabilities = F.softmax(logits, dim=1).numpy()[:, 1] # back to numpy

  # https://stackoverflow.com/questions/69087044/early-stopping-in-bert-trainer-instances
  return {'pr_auc': metrics.pr_auc_score(y_test=labels, probs=probabilities)}

def model_selection(dataset_name, train_dir, learning_rate):
  """Fine-tune on phenotyping data"""

  # deterministic determinism
  torch.manual_seed(2022)
  random.seed(2022)

  model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_path,
    num_labels=2)

  train_files, dev_files = train_test_split(train_dir)
  train_dataset = phenot_data.PhenotypingDataset(train_files, tokenizer_path)
  dev_dataset = phenot_data.PhenotypingDataset(dev_files, tokenizer_path)

  training_args = TrainingArguments(
    output_dir='./Results',
    overwrite_output_dir=True,
    num_train_epochs=model_selection_n_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=learning_rate,
    load_best_model_at_end=True,                 # TODO: change to false
    metric_for_best_model=metric_for_best_model,
    save_strategy=IntervalStrategy.EPOCH,        # TODO: change to no
    evaluation_strategy=IntervalStrategy.EPOCH,
    disable_tqdm=True)
  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics)
  trainer.train()

  best_n_epochs = None
  best_metric_value = trainer.state.best_metric

  print('########## %s ##########' % dataset_name)
  for entry in trainer.state.log_history:
    if metric_for_best_model in entry:
      print('ep: %s, perf: %s' % (entry['epoch'], entry[metric_for_best_model]))
      if entry[metric_for_best_model] == best_metric_value:
        best_n_epochs = entry['epoch']

  print('best epochs: %s, best performance: %s' % (best_n_epochs, best_metric_value))
  return best_n_epochs, best_metric_value

def eval_on_test(
 dataset_name,
 train_dir,
 test_dir,
 optimal_n_epochs,
 optimal_learning_rate):
  """Fine-tune and evaluate on test set"""

  # deterministic determinism
  torch.manual_seed(2022)
  random.seed(2022)

  model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_path,
    num_labels=2)

  train_dataset = phenot_data.PhenotypingDataset(train_dir, tokenizer_path)
  test_dataset = phenot_data.PhenotypingDataset(test_dir, tokenizer_path)

  training_args = TrainingArguments(
    output_dir='./Results',
    overwrite_output_dir=True,
    num_train_epochs=optimal_n_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=optimal_learning_rate,
    load_best_model_at_end=False, # just run for specified num of epochs
    save_strategy=IntervalStrategy.NO,
    evaluation_strategy=IntervalStrategy.NO,
    disable_tqdm=True)
  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset)
  trainer.train()

  predictions = trainer.predict(test_dataset)
  logits = torch.from_numpy(predictions.predictions)
  probabilities = F.softmax(logits, dim=1).numpy()[:, 1]
  labels = np.argmax(logits, axis=1)

  out_file = open(results_file, 'a')
  pr_auc = metrics.report_pr_auc(test_dataset.y, probabilities)
  out_file.write('dataset: %s, prauc: %s\n' % (dataset_name, pr_auc))
  out_file.close()

def main():
  """Evaluate on a few datasets"""

  base_path = os.environ['DATA_ROOT']

  for dataset_name, train_dir, test_dir in eval_datasets:
    train_dir = os.path.join(base_path, train_dir)
    test_dir = os.path.join(base_path, test_dir)

    best_n_epochs, best_metric_value = model_selection(
      dataset_name,
      train_dir,
      learning_rate=lr)

    eval_on_test(
      dataset_name,
      train_dir,
      test_dir,
      optimal_n_epochs=best_n_epochs,
      optimal_learning_rate=lr)

if __name__ == "__main__":
  "My kind of street"

  main()
