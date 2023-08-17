#!/usr/bin/env python3

import transformers, torch, os, pandas, string
from transformers import AutoTokenizer, AutoModelForCausalLM
from time import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

drbench_path = 'DrBench/Csv/summ_0821_test.csv'
model_path = '/home/dima/Lama/Models/Llama-2-13b-chat-hf'

system_prompt = \
  """You are a physician. Please provide a concise summary of problems/diagnoses
  based on the assessment below. Format the output as a bullet-point list."""

def csv_to_assm_sum_tuples():
  """Get summarization input/output pair tuples"""

  data_csv = os.path.join(base_path, drbench_path)
  df = pandas.read_csv(data_csv, dtype='str')

  # input/output pairs
  ios = []

  for assm, sum in zip(df['Assessment'], df['Summary']):

    # sometimes assm is empty and pandas returns a float
    if type(assm) == str and type(sum) == str:
      assm = ''.join(c for c in assm if c in string.printable)
      sum = ''.join(c for c in sum if c in string.printable)
      ios.append((assm, sum))

  return ios

def main(inputs_outputs):
  """Ask for input and feed into llama2"""

  start = time()

  tokenizer = AutoTokenizer.from_pretrained(model_path)
  model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map='auto',
    load_in_8bit=True)
  pipeline = transformers.pipeline(
    'text-generation',
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map='auto')

  end = time()
  print(f'\n[model load time: {end - start} seconds]\n')

  for train_input, train_output in inputs_outputs:
    user_message = train_input
    prompt = f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n\n{user_message} [/INST]\n\n'
    start = time()

    outputs = pipeline(
      prompt,
      do_sample=True,
      top_k=10,
      num_return_sequences=1,
      eos_token_id=tokenizer.eos_token_id,
      temperature=0.001,
      max_length=500)

    end = time()
    for output in outputs:
      print('\n[********** begin generated text **********]\n')
      print(output['generated_text'])
      print('\n[********** end generated text **********]\n')
      print(f'reference summary: {train_output}\n\n')

    print(f'[inference time: {end - start} seconds]\n')
if __name__ == "__main__":

  base_path = os.environ['DATA_ROOT']

  ios = csv_to_assm_sum_tuples()
  main(ios)
