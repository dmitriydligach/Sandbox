#! /usr/bin/env python3
import pandas, string

# Create samples for pretraining transformers from MIMIC notes
# For now each note is one sample, but will break down further later

mimic_notes_file = '/Users/Dima/Work/Data/MimicIII/Source/NOTEEVENTS.csv'
out_file = 'notes.txt'

def extract_notes():
  """Extract only mimic notes to a csv file"""

  outfile = open(out_file, 'w')
  frame = pandas.read_csv(mimic_notes_file, dtype='str', nrows=None)

  for text in frame.TEXT:
    printable = ''.join(c for c in text if c in string.printable)
    printable = printable.replace('\n', '')
    outfile.write(printable + '\n')

if __name__ == "__main__":

  extract_notes()
