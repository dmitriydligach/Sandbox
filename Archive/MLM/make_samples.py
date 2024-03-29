#! /usr/bin/env python3
import pandas, string, os, pathlib, numpy

# Create samples for pretraining transformers from MIMIC notes
# For now each note is one sample, but will break down further later

base = os.environ['DATA_ROOT']

mimic_notes_file = os.path.join(base, 'MimicIII/Source/NOTEEVENTS.csv')
mimic_cui_dir = os.path.join(base, 'MimicIII/Encounters/Cuis/All/')
out_file = 'notes.txt'

def extract_cuis(use_unique_cuis=True):
  """Each line is the content of one cui file"""

  cui_counts = []
  outfile = open(out_file, 'w')

  for cui_file in pathlib.Path(mimic_cui_dir).glob('*.txt'):
    cuis_from_file = pathlib.Path(cui_file).read_text()

    # remove first C from each CUI to avoid tokenization problems
    cui_list = [cui[1:] for cui in cuis_from_file.split()]

    if use_unique_cuis:
      cui_list = list(set(cui_list))

    cui_counts.append(len(cui_list))
    cui_string = ' '.join(cui_list)

    outfile.write(cui_string + '\n')

  print(f'unique cui flag is {use_unique_cuis}')
  print(f'average number of cuis: {numpy.mean(cui_counts):.2f}')
  print(f'median number of cuis: {numpy.median(cui_counts):.2f}')
  print(f'standard deviation: {numpy.std(cui_counts):.2f}')
  print(f'max number of cuis: {numpy.max(cui_counts)}')

def extract_notes():
  """Extract only mimic notes to a csv file"""

  outfile = open(out_file, 'w')
  frame = pandas.read_csv(mimic_notes_file, dtype='str', nrows=None)

  for text in frame.TEXT:
    printable = ''.join(c for c in text if c in string.printable)
    # printable = printable.replace('\n', '') # problem: this merges words from different lines
    outfile.write(printable + '\n')

if __name__ == "__main__":

  extract_notes()
