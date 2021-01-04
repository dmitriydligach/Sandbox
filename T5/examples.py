import argparse

from transformers import T5Tokenizer, T5ForConditionalGeneration

if __name__ == "__main__":
  "My kind of street"

  hparam_dict = dict(model_name='t5-11b')
  hparams = argparse.Namespace(**hparam_dict)
  print('hyper-parameters:', hparams)

  tokenizer = T5Tokenizer.from_pretrained(hparams.model_name)
  model = T5ForConditionalGeneration.from_pretrained(hparams.model_name)

  train_example = 'The <extra_id_0> walks in <extra_id_1> park'
  train_labels = '<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>'

  input_ids = tokenizer(train_example, return_tensors='pt').input_ids
  labels = tokenizer(train_labels, return_tensors='pt').input_ids

  outputs = model(input_ids=input_ids, labels=labels)
  loss = outputs.loss
  logits = outputs.logits

  text = "summarize: state authorities dispatched emergency " \
         "crews tuesday to survey the damage after an onslaught " \
         "of severe weather in mississippi."
  input_ids = tokenizer(text, return_tensors="pt").input_ids
  outputs = model.generate(input_ids)
  print('summary:', tokenizer.decode(outputs.squeeze()))

  s = "translate English to German: one two three four five."
  input_ids = tokenizer(s, return_tensors="pt").input_ids
  outputs = model.generate(input_ids)
  print('translation:', tokenizer.decode(outputs.squeeze()))
