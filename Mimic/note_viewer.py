#! /usr/bin/env python3

import pandas, string, os

base = os.environ['DATA_ROOT']

mimic_notes_file = os.path.join(base, 'MimicIII/Source/NOTEEVENTS.csv')
out_file = 'all-mimic-notes.txt'

def extract_notes():
  """Write all mimic notes to one file for viewing"""

  outfile = open(out_file, 'w')
  frame = pandas.read_csv(mimic_notes_file, dtype='str', nrows=None)

  for id, adm, cat, text in \
   zip(frame.ROW_ID, frame.HADM_ID, frame.CATEGORY, frame.TEXT):

    # remove non-printable characters for ctakes
    printable = ''.join(c for c in text if c in string.printable)

    # print header
    outfile.write('='*75 + '\n')
    outfile.write(f'Admission ID: {adm}\n')
    outfile.write(f'Note ID: {id}\n')
    outfile.write(f'Note type: {cat}\n')
    outfile.write('='*75 + '\n\n')

    outfile.write(printable + '\n\n')

if __name__ == "__main__":

  extract_notes()
