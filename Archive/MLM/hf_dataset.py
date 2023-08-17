#!/usr/bin/env python3

from datasets import load_dataset

def main():
  """See if we can load it"""

  dataset = load_dataset('text', data_files='notes.txt')

  print(dataset)
  print(dataset['train']['text'][50])

if __name__ == "__main__":

  main()
