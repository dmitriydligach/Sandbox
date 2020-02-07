#!/usr/bin/env python3

from cassis import *

type_system_path = './TypeSystem.xml'
xmi_path = './XmiMultViews/ID001_clinic_001.xmi'
event_type = 'org.apache.ctakes.typesystem.type.textsem.EventMention'
sent_type = 'org.apache.ctakes.typesystem.type.textspan.Sentence'

if __name__ == "__main__":

  ts_file = open(type_system_path, 'rb')
  type_system = load_typesystem(ts_file)

  xmi_file = open(xmi_path, 'rb')
  cas = load_cas_from_xmi(xmi_file, typesystem=type_system)
  gold_view = cas.get_view('GoldView')
  sys_view = cas.get_view('_InitialView')

  for sentence in sys_view.select(sent_type):
    print('sentence:', sentence.get_covered_text())

    for event in gold_view.select_covered(event_type, sentence):
      text = event.get_covered_text()
      dtr = event.event.properties.docTimeRel
      print('{} - {}'.format(text, dtr))
    print()
