#!/usr/bin/env python3

from torch.utils.data import Dataset
from datasets import load_dataset

from transformers import T5Tokenizer

# based on a blog post from: https://towardsdatascience.com/
# fine-tuning-a-t5-transformer-for-any-summarization-task-82334c64c81

class WikiHow(Dataset):
  """WikiHow data"""

  def __init__(
   self,
   tokenizer,
   split,
   input_length=512,
   output_length=100):
    """Wikihow train, validation, or test set"""

    self.dataset = load_dataset(
      'wikihow',
      'all',
      data_dir='Data/',
      split=split)

    self.tokenizer = tokenizer
    self.input_length = input_length
    self.output_length = output_length

  def __len__(self):
    """Requried by pytorch"""

    return self.dataset.shape[0]

  def clean_text(self, text):
    """Do we even need this?"""

    text = text.replace('\n', '')
    text = text.replace('``', '')
    text = text.replace('"', '')

    return text

  def convert_to_features(self, instance):
    """Prepare inputs and outputs"""

    text = self.clean_text(instance['text'])
    summary = self.clean_text(instance['headline'])

    text = self.tokenizer(
      [text],
      max_length=self.input_length,
      padding='max_length',
      truncation=True,
      return_tensors='pt')

    summary = self.tokenizer(
      [summary],
      max_length=self.output_length,
      padding='max_length',
      truncation=True,
      return_tensors='pt')

    return text, summary

  def __getitem__(self, index):
    """Required by pytorch"""

    text, summary = self.convert_to_features(self.dataset[index])

    input_ids = text['input_ids'].squeeze()
    input_mask = text['attention_mask'].squeeze()

    output_ids = summary['input_ids'].squeeze()
    output_mask = summary['attention_mask'].squeeze()

    return input_ids, input_mask, output_ids, output_mask

if __name__ == "__main__":
  """My main man"""

  tokenizer = T5Tokenizer.from_pretrained('t5-small')
  dataset = WikiHow(tokenizer, 'validation')
  print('dataset length:', len(dataset))

  data = dataset[50]
  print("instance shape: ", data['source_ids'].shape)
  print("text: ", tokenizer.decode(data['source_ids']))
  print("summary: ", tokenizer.decode(data['target_ids']))
