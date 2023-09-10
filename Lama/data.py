#!/usr/bin/env python3

import os, pandas, string, datasets


drbench_dev_path = 'DrBench/Csv/summ_0821_dev.csv'
drbench_train_path = 'DrBench/Csv/summ_0821_train.csv'

system_prompt = 'You are a physician. Please list the most important ' \
                'problems/diagnoses based on the progress note text ' \
                'below. Only list the problems/diagnoses and nothing else. ' \
                'Be concise.'

def csv_to_fine_tune_data(data_csv_path):
  """Format training data for fine-tuning and make a HF dataset"""

  df = pandas.read_csv(data_csv_path, dtype='str')

  # input/output pairs
  train_samples = []

  for assm, summ, _ in zip(df['Assessment'], df['Summary'], df['S']):

    # sometimes assm is empty and pandas returns a float
    if type(assm) == str and type(summ) == str:

      assm = ''.join(c for c in assm if c in string.printable)
      summ = ''.join(c for c in summ if c in string.printable)
      train_text = f'### Assessment Section ###\n\n{assm}\n\n' \
                   f'### Problem List ###\n\n{summ}'
      train_samples.append(train_text)

  data = datasets.Dataset.from_dict({'text': train_samples})
  # split_data = data.train_test_split(test_size=0.2, shuffle=True)
  # return split_data

  return data

def csv_to_zero_shot_data():
  """Get summarization input/output pair tuples"""

  data_csv = os.path.join(base_path, drbench_dev_path)
  df = pandas.read_csv(data_csv, dtype='str')

  # input/output pairs
  ios = []

  for assm, summ, subj in zip(df['Assessment'], df['Summary'], df['S']):

    # sometimes assm is empty and pandas returns a float
    if type(assm) == str and type(summ) == str and type(subj) == str:

      assm = ''.join(c for c in assm if c in string.printable)
      summ = ''.join(c for c in summ if c in string.printable)
      subj = ''.join(c for c in subj if c in string.printable)

      input_text = f'### Subjective Section ###\n\n{subj}\n\n' \
                   f'### Assessment Section ###\n\n{assm}'
      ios.append((input_text, summ))

  return ios

if __name__ == "__main__":

  base_path = os.environ['DATA_ROOT']
  data_csv_path = os.path.join(base_path, drbench_train_path)

  data = csv_to_fine_tune_data(data_csv_path)
