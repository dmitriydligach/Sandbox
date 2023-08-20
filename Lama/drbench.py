#!/usr/bin/env python3

import transformers, torch, os, pandas, string, numpy
from transformers import AutoTokenizer, AutoModelForCausalLM
from time import time
from rouge_score import rouge_scorer

# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

drbench_path = 'DrBench/Csv/summ_0821_dev.csv'
model_path = '/home/dima/Lama/Models/Llama-2-13b-chat-hf'

system_prompt = \
  """You are a physician. Please provide a concise summary of problems/diagnoses
  based on the assessment below. Format the output as a bullet point list."""

def calc_rougel(generated_text, reference_text):
  """Compute Rouge-L score"""

  # {'rougeL': Score(precision=0.5, recall=0.6, fmeasure=0.5)}
  scorer = rouge_scorer.RougeScorer(['rougeL'])
  scores = scorer.score(reference_text, generated_text)
  f1 = scores['rougeL'].fmeasure

  return f1

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

  f1s = []
  for input_text, reference_output in inputs_outputs:
    prompt = f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n\n{input_text} [/INST]\n\n'

    start = time()
    generated_outputs = pipeline(
      prompt,
      do_sample=True,
      top_k=10,
      num_return_sequences=1,
      eos_token_id=tokenizer.eos_token_id,
      temperature=0.001,
      max_length=500)
    end = time()

    # remove the the prompt from output
    end_index = generated_outputs[0]['generated_text'].index('[/INST]')
    generated_text = generated_outputs[0]['generated_text'][end_index+7:]

    print('\n[********** begin generated text **********]\n')
    print(generated_text)
    print('\n[********** end generated text **********]\n')
    print(f'reference summary: {reference_output}\n\n')
    f1 = calc_rougel(generated_text.lower(), reference_output.lower())
    f1s.append(f1)
    print(f'f1={f1}')

    print(f'[inference time: {end - start} seconds]\n')

  print('average f1:', numpy.mean(f1s))

if __name__ == "__main__":

  base_path = os.environ['DATA_ROOT']

  ios = csv_to_assm_sum_tuples()
  main(ios)
