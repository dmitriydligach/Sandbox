#!/usr/bin/env python3

from cassis import *

if __name__ == "__main__":

  ts_file = open('TypeSystem.xml', 'rb')
  type_system = load_typesystem(ts_file)

  xmi_file = open('XmiSingleView/patientX_doc3_NOTE.txt.xmi', 'rb')
  cas = load_cas_from_xmi(xmi_file, typesystem=type_system)

  for sentence in cas.select('org.apache.ctakes.typesystem.type.textspan.Sentence'):
    print('\nsentence:', sentence.get_covered_text())

    # for token in cas.select_covered('org.apache.ctakes.typesystem.type.syntax.BaseToken', sentence):
    for token in cas.select_covered('org.apache.ctakes.typesystem.type.syntax.WordToken', sentence):
      print('token:', token.get_covered_text())
