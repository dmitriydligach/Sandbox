#!/usr/bin/env python3

import transformers, torch, os, pandas, string, numpy
from transformers import AutoTokenizer, AutoModelForCausalLM
from time import time
from rouge_score import rouge_scorer

lama_size = '7b'
drbench_path = 'DrBench/Csv/summ_0821_dev.csv'
model_path = f'/home/dima/Lama/Models/Llama-2-{lama_size}-chat-hf'

system_prompt = 'You are a physician. Please list the most important ' \
                'problems/diagnoses based on the progress note text ' \
                'below. Only list the problems/diagnoses and nothing else. ' \
                'Be concise.'

if '7b' in model_path:
  os.environ['CUDA_VISIBLE_DEVICES'] = '0'
elif '13b' in model_path:
  os.environ['CUDA_VISIBLE_DEVICES'] = '1'
elif '70b' in model_path:
  os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'

def calc_rougel(generated_text, reference_text):
  """Compute Rouge-L score"""

  # {'rougeL': Score(precision=0.5, recall=0.6, fmeasure=0.5)}
  scorer = rouge_scorer.RougeScorer(['rougeL'])
  scores = scorer.score(reference_text, generated_text)
  f1 = scores['rougeL'].fmeasure

  return f1

def csv_to_io_tuples():
  """Get summarization input/output pair tuples"""

  data_csv = os.path.join(base_path, drbench_path)
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

def main(inputs_outputs):
  """Ask for input and feed into llama2"""

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

  f1s = []
  inference_times = []

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
      max_length=2000)

    end = time()
    inference_times.append(end - start)

    print('\n********** begin generated text **********\n')
    print(generated_outputs[0]['generated_text'])
    print('\n********** end generated text **********\n')

    print(f'reference summary: {reference_output}\n\n')

    # remove the the prompt from output and evaluate
    end_index = generated_outputs[0]['generated_text'].index('[/INST]')
    generated_text = generated_outputs[0]['generated_text'][end_index+7:]
    f1 = calc_rougel(generated_text.lower(), reference_output.lower())
    f1s.append(f1)

  av_inf_time = numpy.mean(inference_times)
  print(f'\naverage inference time: {av_inf_time} seconds]')
  print('average f1:', numpy.mean(f1s))

if __name__ == "__main__":

  base_path = os.environ['DATA_ROOT']

  inputs_and_outputs = csv_to_io_tuples()

  main(inputs_and_outputs)
