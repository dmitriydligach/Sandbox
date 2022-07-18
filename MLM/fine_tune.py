#!/usr/bin/env python3

import torch, random, phenot_data, os, metrics
import numpy as np
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from transformers import (TrainingArguments,
                          Trainer,
                          AutoModelForSequenceClassification)

from datasets import load_metric

# deterministic determinism
torch.manual_seed(2022)
random.seed(2022)

def main():
  """Fine-tune on phenotyping data"""

  model = AutoModelForSequenceClassification.from_pretrained(
    'Output',
    num_labels=2)

  train_dataset = phenot_data.PhenotypingDataset(train_dir, tok_path)
  test_dataset = phenot_data.PhenotypingDataset(test_dir, tok_path)

  training_args = TrainingArguments(
    output_dir='./Results',
    num_train_epochs=10,
    per_device_train_batch_size=48,
    per_device_eval_batch_size=48,
    learning_rate=5e-5,
    load_best_model_at_end=True,
    save_strategy='no',
    evaluation_strategy='no',
    disable_tqdm=True)
  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset)
  trainer.train()

  predictions = trainer.predict(test_dataset)
  logits = torch.from_numpy(predictions.predictions)
  probabilities = F.softmax(logits, dim=1).numpy()[:, 1]
  labels = np.argmax(logits, axis=1)

  print('*** evaluation results ***')
  metrics.report_accuracy(test_dataset.y, labels)
  metrics.report_roc_auc(test_dataset.y, probabilities)
  metrics.report_pr_auc(test_dataset.y, probabilities)

if __name__ == "__main__":
  "My kind of street"

  base = os.environ['DATA_ROOT']
  train_dir = os.path.join(base, 'Opioids1k/Train/')
  test_dir = os.path.join(base, 'Opioids1k/Test/')
  tok_path = 'Tokenizer'

  main()
