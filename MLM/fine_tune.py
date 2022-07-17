#!/usr/bin/env python3

import torch, random, phenot_data, os, metrics
import numpy as np
from transformers import (TrainingArguments,
                          Trainer,
                          AutoModelForSequenceClassification)

from datasets import load_metric
metric = load_metric("accuracy")

# deterministic determinism
torch.manual_seed(2022)
random.seed(2022)

# def compute_metrics(eval_pred):
#   """Metric"""
#
#   logits, labels = eval_pred
#   predictions = np.argmax(logits, axis=-1)
#
#   return metric.compute(predictions=predictions, references=labels)

def main():
  """Fine-tune on phenotyping data"""

  model = AutoModelForSequenceClassification.from_pretrained(
    'Output',
    num_labels=2)

  train_dataset = phenot_data.PhenotypingDataset(train_dir, tok_path)
  test_dataset = phenot_data.PhenotypingDataset(test_dir, tok_path)

  training_args = TrainingArguments(
    output_dir='./Results',
    num_train_epochs=25,
    per_device_train_batch_size=48,
    per_device_eval_batch_size=48,
    learning_rate=1e-5,
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

  results = trainer.predict(test_dataset)
  test_predictions = np.argmax(results.predictions, axis=1)

  print()
  metrics.report_accuracy(test_dataset.y, test_predictions)

if __name__ == "__main__":
  "My kind of street"

  base = os.environ['DATA_ROOT']
  train_dir = os.path.join(base, 'Opioids1k/Train/')
  test_dir = os.path.join(base, 'Opioids1k/Test/')
  tok_path = 'Tokenizer'

  main()
