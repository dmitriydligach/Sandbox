#!/usr/bin/env python3

import os, glob
from cassis import *

section_type = 'webanno.custom.Sectionizer'
assessment_plan_type = 'webanno.custom.AssessmentPlanLink'
xmi_file_path = '/Users/Dima/Work/Wisconsin/AssessAndPlan/Ryan_Inception_Export/101125.xmi'
type_system_path = '/Users/Dima/Work/Wisconsin/AssessAndPlan/Ryan_Inception_Export/TypeSystem.xml'


class color:
  PURPLE = '\033[95m'
  CYAN = '\033[96m'
  DARKCYAN = '\033[36m'
  BLUE = '\033[94m'
  GREEN = '\033[92m'
  YELLOW = '\033[93m'
  RED = '\033[91m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'
  END = '\033[0m'

def assessment_plan_link():
  """Extract events and times"""

  xmi_file = open(xmi_file_path, 'rb')
  type_system_file = open(type_system_path, 'rb')
  type_system = load_typesystem(type_system_file)
  cas = load_cas_from_xmi(xmi_file, typesystem=type_system)
  sys_view = cas.get_view('_InitialView')

  # iterate over sentences extracting events and times
  for section in sys_view.select(assessment_plan_type):

    if(section.next):
      print('RELATION TYPE:', section.referenceRelation)
      print('-' * 100)

      print('SOURCE SECTION TYPE:', section.next.referenceType)
      print('SOURCE SECTION TEXT:\n', section.next.get_covered_text())
      print('-' * 100)

      print('TARGET SECTION TYPE:', section.referenceType)
      print('TARGET SECTION TEXT:\n', section.get_covered_text())
      print('='*100)

if __name__ == "__main__":
  """My main man"""

  assessment_plan_link()