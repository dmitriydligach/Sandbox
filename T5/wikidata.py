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
    """Not sure what the first two lines do"""

    text = text.replace('Example of text:', '')
    text = text.replace('Example of Summary:', '')
    text = text.replace('\n', '')
    text = text.replace('``', '')
    text = text.replace('"', '')

    return text

  def convert_to_features(self, instance):
    """Tokenize contexts and questions (as pairs of inputs)"""

    input_ = self.clean_text(instance['text'])
    target_ = self.clean_text(instance['headline'])

    source = self.tokenizer.batch_encode_plus(
      [input_],
      max_length=self.input_length,
      padding='max_length',
      truncation=True,
      return_tensors='pt')

    targets = self.tokenizer.batch_encode_plus(
      [target_],
      max_length=self.output_length,
      padding='max_length',
      truncation=True,
      return_tensors='pt')

    return source, targets

  def __getitem__(self, index):
    """Required by pytorch"""

    source, targets = self.convert_to_features(self.dataset[index])

    source_ids = source["input_ids"].squeeze()
    target_ids = targets["input_ids"].squeeze()

    source_mask = source["attention_mask"].squeeze()
    target_mask = targets["attention_mask"].squeeze()

    # return {
    #   "source_ids": source_ids,
    #   "source_mask": source_mask,
    #   "target_ids": target_ids,
    #   "target_mask": target_mask}

    return source_ids, source_mask, target_ids, target_mask

if __name__ == "__main__":
  """My main man"""

  tokenizer = T5Tokenizer.from_pretrained('t5-small')
  dataset = WikiHow(tokenizer, 'validation')
  print('dataset length:', len(dataset))

  data = dataset[50]
  print("instance shape: ", data['source_ids'].shape)
  print("text: ", tokenizer.decode(data['source_ids']))
  print("summary: ", tokenizer.decode(data['target_ids']))
