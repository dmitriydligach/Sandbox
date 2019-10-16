#!/usr/bin/env python3

import torch
from transformers import BertTokenizer, BertConfig, AdamW
from transformers import BertForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
import numpy as np
import glob, os, logging

logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

# imdb training data (ignoring test for now)
data_pos = '/home/dima/Data/Imdb/train/pos/*.txt'
data_neg = '/home/dima/Data/Imdb/train/neg/*.txt'

# hyper-parameters
max_files = all
max_len = 512
batch_size = 8
epochs = 4

def load_data():
  """Rotten tomatoes"""

  labels = []
  sentences = []

  for file in glob.glob(data_pos)[:max_files]:
    labels.append(1)
    sentences.append('[CLS] ' + open(file).read()[:max_len] + ' [SEP]')
  for file in glob.glob(data_neg)[:max_files]:
    labels.append(0)
    sentences.append('[CLS] ' + open(file).read()[:max_len] + ' [SEP]')

  return sentences, labels

def flat_accuracy(preds, labels):
  """Calculate the accuracy of our predictions vs labels"""

  pred_flat = np.argmax(preds, axis=1).flatten()
  labels_flat = labels.flatten()

  return np.sum(pred_flat == labels_flat) / len(labels_flat)

if __name__ == "__main__":

  # deal with warnings for now
  os.system('clear')

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  n_gpu = torch.cuda.device_count()
  torch.cuda.get_device_name(0)

  sentences, labels = load_data()
  print('loaded %d examples and %d labels...' % (len(sentences), len(labels)))

  # tokenize and convert to ints
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
  tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
  input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
  input_ids = pad_sequences(input_ids, maxlen=max_len, dtype="long", truncating="post", padding="post")

  # create attention masks
  attention_masks = []
  for seq in input_ids:
    # use 1s for tokens and 0s for padding
    seq_mask = [float(i > 0) for i in seq]
    attention_masks.append(seq_mask)

  # make validation set
  train_inputs, validation_inputs, train_labels, validation_labels = \
    train_test_split(input_ids, labels, test_size=0.1, random_state=0)
  train_masks, validation_masks, _, _ = \
    train_test_split(attention_masks, input_ids, test_size=0.1, random_state=0)

  # convert everything into torch tensors
  train_inputs = torch.tensor(train_inputs)
  validation_inputs = torch.tensor(validation_inputs)
  train_labels = torch.tensor(train_labels)
  validation_labels = torch.tensor(validation_labels)
  train_masks = torch.tensor(train_masks)
  validation_masks = torch.tensor(validation_masks)

  # create iterators for our data
  train_data = TensorDataset(train_inputs, train_masks, train_labels)
  train_sampler = RandomSampler(train_data)
  train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
  validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
  validation_sampler = SequentialSampler(validation_data)
  validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

  # load pretrained bert model with a single linear classification layer on top
  model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
  model.cuda()

  param_optimizer = list(model.named_parameters())
  no_decay = ['bias', 'gamma', 'beta']
  optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}]

  # this variable contains all of the hyperparemeter information our training loop needs
  optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)

  # store our loss and accuracy for plotting
  train_loss_set = []

  # training loop
  for _ in trange(epochs, desc="Epoch"):

    # Set our model to training mode (as opposed to evaluation mode)
    model.train()

    # Tracking variables
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0

    # train for one epoch
    for step, batch in enumerate(train_dataloader):
    
      # add batch to GPU
      batch = tuple(t.to(device) for t in batch)
    
      # unpack the inputs from our dataloader
      b_input_ids, b_input_mask, b_labels = batch
    
      # clear out the gradients (by default they accumulate)
      optimizer.zero_grad()
    
      # forward pass
      loss, logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
      train_loss_set.append(loss.item())
          
      # backward pass
      loss.backward()
          
      # update parameters and take a step using the computed gradient
      optimizer.step()

      # update tracking variables
      tr_loss += loss.item()
      nb_tr_examples += b_input_ids.size(0)
      nb_tr_steps += 1

    print("train loss: {}".format(tr_loss/nb_tr_steps))

    # put model in evaluation mode to evaluate loss on the validation set
    model.eval()

    # tracking variables
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    # evaluate data for one epoch
    for batch in validation_dataloader:
      
      # add batch to GPU
      batch = tuple(t.to(device) for t in batch)
      
      # unpack the inputs from our dataloader
      b_input_ids, b_input_mask, b_labels = batch
      
      # don't compute or store gradients
      with torch.no_grad():
        # forward pass; only logits returned since labels not provided
        [logits] = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

      # move logits and labels to CPU
      logits = logits.detach().cpu().numpy()
      label_ids = b_labels.to('cpu').numpy()

      tmp_eval_accuracy = flat_accuracy(logits, label_ids)

      eval_accuracy += tmp_eval_accuracy
      nb_eval_steps += 1

    print("validation Accuracy: {}\n".format(eval_accuracy/nb_eval_steps))
