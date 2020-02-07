#!/usr/bin/env python3

from cassis import *

type_system_path = './TypeSystem.xml'
xmi_path = './XmiMultViews/ID001_clinic_001.xmi'
type = 'org.apache.ctakes.typesystem.type.textsem.EventMention'

if __name__ == "__main__":

  ts_file = open(type_system_path, 'rb')
  type_system = load_typesystem(ts_file)

  xmi_file = open(xmi_path, 'rb')
  cas = load_cas_from_xmi(xmi_file, typesystem=type_system)
  gold_view = cas.get_view("GoldView")

  for event in gold_view.select(type):
    print(event.get_covered_text())
    print('misc:', event.event.properties.docTimeRel)
    print()
