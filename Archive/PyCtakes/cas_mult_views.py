#!/usr/bin/env python3

from cassis import *

if __name__ == "__main__":

  ts_file = open('TypeSystem.xml', 'rb')
  type_system = load_typesystem(ts_file)

  xmi_file = open('XmiMultViews/ID001_clinic_001.xmi', 'rb')
  cas = load_cas_from_xmi(xmi_file, typesystem=type_system)
  gold_view = cas.get_view("GoldView")

  for event in gold_view.select('org.apache.ctakes.typesystem.type.textsem.EventMention'):
    print(event.get_covered_text())
