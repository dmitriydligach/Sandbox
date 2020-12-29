#!/usr/bin/env python3

from torch.utils.data import Dataset
from datasets import load_dataset

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)

# https://towardsdatascience.com/fine-tuning-a-t5-transformer-for-any-summarization-task-82334c64c81

class WikiHow(Dataset):
  """WikiHow data"""

  def __init__(self, tokenizer, type_path, num_samples, input_length, output_length, print_text=False):
    """Constructing the deconstruction"""

    self.dataset = load_dataset(
      'wikihow',
      'all',
      data_dir='Data/',
      split=type_path)

    if num_samples:
      self.dataset = self.dataset.select(list(range(0, num_samples)))

    self.input_length = input_length
    self.tokenizer = tokenizer
    self.output_length = output_length
    self.print_text = print_text

  def __len__(self):
    """Requried by pytorch"""

    return self.dataset.shape[0]

  def clean_text(self, text):
    """Not sure what the first two lines do"""

    text = text.replace('Example of text:', '')
    text = text.replace('Example of Summary:', '')
    text = text.replace('\n', '')
    text = text.replace('``', '')
    text = text.replace('"', '')

    return text

  def convert_to_features(self, example_batch):
    """Tokenize contexts and questions (as pairs of inputs)"""

    if self.print_text:
      print("Input Text: ", self.clean_text(example_batch['text']))

    input_ = self.clean_text(example_batch['text'])
    target_ = self.clean_text(example_batch['headline'])

    source = self.tokenizer.batch_encode_plus(
      [input_],
      max_length=self.input_length,
      padding='max_length',
      truncation=True,
      return_tensors="pt")

    targets = self.tokenizer.batch_encode_plus(
      [target_],
      max_length=self.output_length,
      padding='max_length',
      truncation=True,
      return_tensors="pt")

    return source, targets

  def __getitem__(self, index):
    """Required by pytorch"""

    source, targets = self.convert_to_features(self.dataset[index])

    source_ids = source["input_ids"].squeeze()
    target_ids = targets["input_ids"].squeeze()

    src_mask = source["attention_mask"].squeeze()
    target_mask = targets["attention_mask"].squeeze()

    return {
      "source_ids": source_ids,
      "source_mask": src_mask,
      "target_ids": target_ids,
      "target_mask": target_mask}

if __name__ == "__main__":
  """My main man"""

  tokenizer = T5Tokenizer.from_pretrained('t5-small')
  dataset = WikiHow(tokenizer, 'validation', None, 512, 150, True)
  print('dataset length:', len(dataset))

  data = dataset[50]
  print("Shape of Tokenized Text: ", data['source_ids'].shape)
  print("Sanity check - Decode Text: ", tokenizer.decode(data['source_ids']))
  print("Sanity check - Decode Summary: ", tokenizer.decode(data['target_ids']))
