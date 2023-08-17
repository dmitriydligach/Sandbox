#!/usr/bin/env python3

import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM

import logging
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
  
  text = "[CLS] Patient had a heart attack in April of 2015. [SEP] He is doing well now . [SEP]"

  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  tokenized_text = tokenizer.tokenize(text)

  print('tokenized text:', tokenized_text)

  masked_index1 = 7
  masked_index2 = 9
  # tokenized_text[masked_index1] = '[MASK]'
  tokenized_text[masked_index2] = '[MASK]'
  print('with a mask:', tokenized_text)

  indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
  segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
  
  tokens_tensor = torch.tensor([indexed_tokens])
  segments_tensors = torch.tensor([segments_ids])

  model = BertForMaskedLM.from_pretrained('bert-base-uncased')
  model.eval()

  with torch.no_grad():
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    predictions = outputs[0]
    
  # predicted_index = torch.argmax(predictions[0, masked_index1]).item()
  # predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
  # print('prediction:', predicted_token)

  predicted_index = torch.argmax(predictions[0, masked_index2]).item()
  predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
  print('prediction:', predicted_token)
  

