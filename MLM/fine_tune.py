#!/usr/bin/env python3

import torch, random, phenot_data, os, pathlib, metrics, shutil
from torch.nn import functional as F
from transformers import (TrainingArguments,
                          Trainer,
                          AutoModelForSequenceClassification,
                          IntervalStrategy)

# misc constants
pretrained_model_path = '/home/dima/Models/CuiBert512/checkpoint-250000/'
output_dir = './Results'
metric_for_best_model = 'eval_pr_auc'
tokenizer_path = './CuiTokenizer'
results_file = './results.txt'

# hyperparameters
model_selection_n_epochs = 25
batch_size = 48

# search over these hyperparameters
classifier_dropouts = [0.1, 0.25, 0.5]
learning_rates = [2e-5, 3e-5, 5e-5, 7e-5]

# datasets
eval_datasets = [('alcohol', 'Alcohol/anc_notes_cuis/', 'Alcohol/anc_notes_test_cuis/'),
                 ('ards', 'Ards/Train/', 'Ards/Test/'),
                 ('injury', 'Injury/Train/', 'Injury/Test/'),
                 ('opioids', 'Opioids1k/Train/', 'Opioids1k/Test/')]

class WeightedTrainer(Trainer):
  """ Need this to compute weighted loss"""

  def __init__(
    self,
    model,
    args,
    train_dataset,
    eval_dataset,
    compute_metrics,
    weights):
    """Deconstruct the constructor"""

    super(WeightedTrainer, self).__init__(
      model=model,
      args=args,
      train_dataset=train_dataset,
      eval_dataset=eval_dataset,
      compute_metrics=compute_metrics)

    self.weights = weights

  def compute_loss(self, model, inputs, return_outputs=False):
    """Weighted loss"""

    labels = inputs.get("labels")

    # forward pass
    outputs = model(**inputs)
    logits = outputs.get("logits")

    # compute weighted loss
    weights = self.weights.to(labels.device)
    loss_fct = torch.nn.CrossEntropyLoss(weights)
    loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))

    return (loss, outputs) if return_outputs else loss

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

def grid_search(dataset_name, train_dir):
  """Try different hyperparameter combinations and return the best"""

  # key: performance, value: [best num of epochs, learning rate]
  search_results = {}

  for classifier_dropout in classifier_dropouts:
    for learning_rate in learning_rates:
      print('evaluating lr and dropout:', classifier_dropout, learning_rate)
      best_n_epochs, best_metric_value = eval_on_dev_set(
        dataset_name,
        train_dir,
        learning_rate,
        classifier_dropout)
      search_results[best_metric_value] = \
        [best_n_epochs, learning_rate, classifier_dropout]

  print('search results:', dataset_name, search_results)
  best_performance = max(search_results.keys())
  print('best performance:', best_performance)
  optimal_n_epochs, optimal_learning_rate, optimal_classifier_dropout = \
    search_results[best_performance]
  print('optimal epochs, lr, dropout:',
    optimal_n_epochs, optimal_learning_rate, optimal_classifier_dropout)

  return optimal_n_epochs, optimal_learning_rate, optimal_classifier_dropout

def eval_on_dev_set(dataset_name, train_dir, learning_rate, classifier_dropout):
  """Make a dev set, fine-tune, and evaluate on it"""

  # deterministic determinism
  torch.manual_seed(2022)
  random.seed(2022)

  model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_path,
    num_labels=2)
  model.dropout = torch.nn.modules.dropout.Dropout(classifier_dropout)

  train_files, dev_files = train_test_split(train_dir)
  train_dataset = phenot_data.PhenotypingDataset(train_files, tokenizer_path)
  dev_dataset = phenot_data.PhenotypingDataset(dev_files, tokenizer_path)

  label_counts = torch.bincount(torch.IntTensor(train_dataset.y))
  weights = len(train_dataset.y) / (2.0 * label_counts)

  training_args = TrainingArguments(
    output_dir=output_dir,
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
  trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    compute_metrics=compute_metrics,
    weights=weights)
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

  # remove intermediate checkpoint dir to save space
  shutil.rmtree(output_dir)

  return best_n_epochs, best_metric_value

def eval_on_test_set(
  dataset_name,
  train_dir,
  test_dir,
  optimal_n_epochs,
  optimal_learning_rate,
  optimal_classifier_dropout):
  """Fine-tune and evaluate on test set"""

  # deterministic determinism
  torch.manual_seed(2022)
  random.seed(2022)

  model = AutoModelForSequenceClassification.from_pretrained(
    pretrained_model_path,
    num_labels=2)
  model.dropout = torch.nn.modules.dropout.Dropout(optimal_classifier_dropout)

  train_dataset = phenot_data.PhenotypingDataset(train_dir, tokenizer_path)
  test_dataset = phenot_data.PhenotypingDataset(test_dir, tokenizer_path)

  label_counts = torch.bincount(torch.IntTensor(train_dataset.y))
  weights = len(train_dataset.y) / (2.0 * label_counts)

  training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=optimal_n_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    learning_rate=optimal_learning_rate,
    load_best_model_at_end=False, # just run for specified num of epochs
    save_strategy=IntervalStrategy.NO,
    evaluation_strategy=IntervalStrategy.NO,
    disable_tqdm=True)
  trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=None,
    compute_metrics=None,
    weights=weights)
  trainer.train()

  predictions = trainer.predict(test_dataset)
  logits = torch.from_numpy(predictions.predictions)
  probabilities = F.softmax(logits, dim=1).numpy()[:, 1]
  # labels = np.argmax(logits, axis=1)

  # remove intermediate checkpoint dir to save space
  shutil.rmtree(output_dir)

  out_file = open(results_file, 'a')
  pr_auc = metrics.report_pr_auc(test_dataset.y, probabilities)
  out_file.write('model: %s, dataset: %s, prauc: %s\n' % (
    pretrained_model_path.split('/')[-2], dataset_name, pr_auc))
  out_file.close()

def main():
  """Evaluate on a few datasets"""

  base_path = os.environ['DATA_ROOT']

  for dataset_name, train_dir, test_dir in eval_datasets:
    train_dir = os.path.join(base_path, train_dir)
    test_dir = os.path.join(base_path, test_dir)

    optimal_n_epochs, optimal_learning_rate, optimal_classifier_dropout = \
      grid_search(dataset_name, train_dir)

    eval_on_test_set(
      dataset_name,
      train_dir,
      test_dir,
      optimal_n_epochs=optimal_n_epochs,
      optimal_learning_rate=optimal_learning_rate,
      optimal_classifier_dropout=optimal_classifier_dropout)

if __name__ == "__main__":
  "My kind of street"

  main()
