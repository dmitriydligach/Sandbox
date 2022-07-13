#!/usr/bin/env python3

import torch, random, phenot_data, os
from transformers import TrainingArguments, Trainer, AutoModelForSequenceClassification

# deterministic determinism
torch.manual_seed(2022)
random.seed(2022)

def main():
  """Fine-tune on phenotyping data"""

  model = AutoModelForSequenceClassification.from_pretrained('Output')

  train_dataset = phenot_data.PhenotypingDataset(train_dir, tok_path)
  test_dataset = phenot_data.PhenotypingDataset(test_dir, tok_path)

  training_args = TrainingArguments(
    output_dir='./Results',
    num_train_epochs=2,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    learning_rate=1e-5,
    disable_tqdm=True)

  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset)

  trainer.train()

if __name__ == "__main__":
  "My kind of street"

  base = os.environ['DATA_ROOT']
  train_dir = os.path.join(base, 'Opioids1k/Train/')
  test_dir = os.path.join(base, 'Opioids1k/Test/')
  tok_path = 'Tokenizer'

  main()
