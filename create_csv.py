import json
import csv
import re

f = open('testomatTests.json')
file = open('tests.csv', 'w')

data = json.load(f)
writer = csv.writer(file)

for test in data:
  title = re.sub(',', '', test['title'])
  title = re.sub('\n', '', title)
  file.write(title + '\n')

f.close()
file.close()

