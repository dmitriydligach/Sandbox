#!/usr/bin/env python3

import os, glob
from cassis import *

section_type = 'webanno.custom.Sectionizer'
assessment_plan_type = 'webanno.custom.AssessmentPlanLink'
xmi_file_path = '/Users/Dima/Work/Wisconsin/AssessAndPlan/Ryan_Inception_Export/101125.xmi'
type_system_path = '/Users/Dima/Work/Wisconsin/AssessAndPlan/Ryan_Inception_Export/TypeSystem.xml'

def assessment_plan_link():
  """Extract events and times"""

  xmi_file = open(xmi_file_path, 'rb')
  type_system_file = open(type_system_path, 'rb')
  type_system = load_typesystem(type_system_file)
  cas = load_cas_from_xmi(xmi_file, typesystem=type_system)
  sys_view = cas.get_view('_InitialView')

  # iterate over sentences extracting events and times
  for section in sys_view.select(assessment_plan_type):

    # there's a single assessment per note
    if(section.referenceType == 'Assessment'):
      print('-' * 75)
      print('ASSESSMENT SECTION')
      print('-' * 75)
      print(section.get_covered_text())
      print('=' * 75)

    # there are multiple treatment plans per note
    else:
      print('TREATMENT PLAN SECTION')
      print('-' * 75)
      print(section.get_covered_text())
      print('-' * 75)
      print('RELATION TO ASSESSMENT SECTION:', section.referenceRelation)
      print('=' * 75)

if __name__ == "__main__":
  """My main man"""

  assessment_plan_link()