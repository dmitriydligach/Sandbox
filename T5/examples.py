from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained('t5-11b')
model = T5ForConditionalGeneration.from_pretrained('t5-11b')

input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1> park', return_tensors='pt').input_ids
labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the <extra_id_2> </s>', return_tensors='pt').input_ids

outputs = model(input_ids=input_ids, labels=labels)
loss = outputs.loss
logits = outputs.logits

s = "summarize: state authorities dispatched emergency crews tuesday to survey the damage after an onslaught of severe weather in mississippi."
input_ids = tokenizer(s, return_tensors="pt").input_ids
outputs = model.generate(input_ids)
tokenizer.convert_ids_to_tokens(outputs.squeeze())

s = "translate English to German: one two three four five."
input_ids = tokenizer(s, return_tensors="pt").input_ids
outputs = model.generate(input_ids)
tokenizer.convert_ids_to_tokens(outputs.squeeze())
