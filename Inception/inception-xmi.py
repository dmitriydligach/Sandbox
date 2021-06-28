#!/usr/bin/env python3

import os, glob
from cassis import *

section_type = 'webanno.custom.Sectionizer'
assessment_plan_type = 'webanno.custom.AssessmentPlanLink'
xmi_files_path = '/Users/Dima/Work/Wisconsin/AssessAndPlan/Ryan_Inception_Export/*.xmi'
type_system_path = '/Users/Dima/Work/Wisconsin/AssessAndPlan/Ryan_Inception_Export/TypeSystem.xml'

def sections():
  """Extract events and times"""

  for xmi_path in glob.glob(xmi_files_path):
    print('xmi file:', xmi_path)
    xmi_file = open(xmi_path, 'rb')
    type_system_file = open(type_system_path, 'rb')
    type_system = load_typesystem(type_system_file)
    cas = load_cas_from_xmi(xmi_file, typesystem=type_system)
    sys_view = cas.get_view('_InitialView')

    # iterate over sentences extracting events and times
    for section in sys_view.select(section_type):
      print('sec: {}, start: {}, end: {}'.format(section.Section, section.begin, section.end))
      print(section.get_covered_text())
      print('='*100)

    # just look at the first xmi file for now
    break

def assessment_plan_link():
  """Extract events and times"""

  for xmi_path in glob.glob(xmi_files_path):
    print('xmi file:', xmi_path)
    xmi_file = open(xmi_path, 'rb')
    type_system_file = open(type_system_path, 'rb')
    type_system = load_typesystem(type_system_file)
    cas = load_cas_from_xmi(xmi_file, typesystem=type_system)
    sys_view = cas.get_view('_InitialView')

    # iterate over sentences extracting events and times
    for section in sys_view.select(assessment_plan_type):
      if(section.next):
        print('reference type:', section.referenceType)
        print('reference relation:', section.referenceRelation)
        print(section.get_covered_text())
        print('-'*50)
        print(section.next.get_covered_text())
        print('='*100)

    # just look at the first xmi file for now
    break

if __name__ == "__main__":
  """My main man"""

  assessment_plan_link()