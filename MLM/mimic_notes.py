#! /usr/bin/env python3
import pandas, string

mimic_notes_file = '/Users/Dima/Work/Data/MimicIII/Source/NOTEEVENTS.csv'
out_file = 'notes.csv'

def extract_notes():
  """Extract only mimic notes to a csv file"""

  outfile = open(out_file, 'w')
  outfile.write('text\n')

  frame = pandas.read_csv(mimic_notes_file, dtype='str', nrows=1000)

  for text in frame.TEXT:
    printable = ''.join(c for c in text if c in string.printable)
    printable = printable.replace('\n', '')
    outfile.write(printable + '\n')

if __name__ == "__main__":

  extract_notes()
