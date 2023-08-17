#!/usr/bin/env python3

import transformers, torch, os
from transformers import AutoTokenizer, AutoModelForCausalLM
from time import time

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_path = '/home/dima/Lama/Models/Llama-2-7b-chat-hf'
system_prompt = 'You are a helpful AI assistant. You always give precise and short answers.'

def main():
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

  while (True):

    user_message = input('Prompt: ')
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
      print('\n[***** begin generated text *****]\n')
      print(output['generated_text'])
      print('\n[***** End generated text *****]\n')

    print(f'[inference time: {end - start} seconds]\n')
if __name__ == "__main__":
   
  main()
