#!/usr/bin/env python3

import os, datetime, shutil

root_dir = '/Users/Dima/GDrive/UTC/'
content_dir = 'Top Level/Engagement Team/Ukraine Teams Content/'
target_dir = '/Users/Dima/Temp/All Media/'

old_dir = os.path.join(root_dir, content_dir)
new_dir = os.path.join(root_dir, target_dir)

# set of allowed media file extensions
media_files = {'jpeg', 'mp4', 'jpg', 'mov'}

# team leaders
team_names = ['Natalia', 'Kseniia', 'Pavel', 'Andriy',
              'Dina', 'Karina', 'Nikolay', 'Timur', 'Olga']

def main():
  """Iterate the hell out of it"""

  for path, dirs, files in os.walk(old_dir):
    if 'Week of' not in path:
      continue # only look at subfolders like 'Week of March 21-March 27'

    for file_name in files:
      orig_path = os.path.join(path, file_name)
      file_name = os.path.basename(orig_path)
      elements = file_name.split('.') # sometimes there are multiple '.'s
      old_name = elements[0]
      extension = elements[-1]

      if extension.lower() not in media_files:
        continue

      # extract team name from the path
      team = None
      for team_name in team_names:
        if team_name in orig_path:
          team = team_name
          break

      if not team:
        print('team name not found:', orig_path)
        continue

      mod_time = os.path.getmtime(orig_path)
      human_time = datetime.datetime.fromtimestamp(mod_time)
      month_name = human_time.strftime("%B")
      # date_only = human_time.strftime('%B-%d-%Y') # use April, May
      date_only = human_time.strftime('%m-%d-%Y')   # use 04, 05

      # create month directory if necessary
      month_dir_path = os.path.join(new_dir, month_name)
      if not os.path.isdir(month_dir_path):
        os.mkdir(month_dir_path)

      # naming convention: <month name>/<date><team><old name>.<extension>
      new_path = '%s/%s-%s-%s.%s' % (month_dir_path, date_only, team, old_name, extension)
      print(orig_path)
      print(new_path)
      print('=' * 120)

      try:
        shutil.copy(orig_path, new_path)
      except:
        print('failed to copy:', orig_path)

if __name__ == "__main__":

  main()