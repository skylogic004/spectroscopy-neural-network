#!/usr/bin/env python3
""" Helps create a unique output directory.
"""

from __future__ import division, print_function

import os
import datetime
import re
from os.path import join, split, realpath, dirname, splitext, isfile, exists, isdir
from os import listdir
import posixpath

TASK_KEY_STR = '_{{{taskKey:s}}}'
CUSTOM_TEXT_DELIMITER = '~'

TASK_RUN_ID_STR = '[{num:06d}]'
BIG_ID_DIR_STR = TASK_RUN_ID_STR+'{taskKey_text:s}{date:s}{customText:s}'
BIG_ID_START = 10000

def newDir(parentOutDir, name=None, id_fpath=None, taskKey=None, use_date=False):
	""" Creates a new directory with the following: a number (ID) that increments, a date, and a name.
	ID can be based on existing directories or on a file that tracks the last used ID.
	"""

	# setup output directory
	if (not os.path.exists(parentOutDir)):
		print('Parent directory (%s) doesn\'t exist, creating it.' % parentOutDir)
		os.makedirs(parentOutDir)

	if (id_fpath is None):
		return simpleNewDir(parentOutDir, name)
	else: # Use a file to store the incrementing ID (so that it can be global)
		if (not os.path.exists(id_fpath)):
			# ID file doesn't exist yet, initialize to some starting ID
			new_id = BIG_ID_START
		else:
			# Read ID file
			with open(id_fpath, mode='r') as f:
				old_id = f.read()
				new_id = int(old_id) + 1

		# Keep track of the ID in id_fpath file
		with open(id_fpath, mode='w') as f:
			f.write(str(new_id))

		if (use_date):
			now = str(datetime.datetime.now()).replace(':','.').replace(' ', '_')
			now = now[:10] # date only, no time
			date_text = '_' + now
		else:
			date_text = ''

		if (name is not None):
			customText = CUSTOM_TEXT_DELIMITER + name
		else:
			customText = ''

		if (taskKey is not None):
			taskKey_text = TASK_KEY_STR.format(taskKey=taskKey)
		else:
			taskKey_text = ''

		outDirName = BIG_ID_DIR_STR.format(
			num=new_id, 
			date=date_text, 
			customText=customText,
			taskKey_text=taskKey_text,
		)
		outDir = posixpath.join(parentOutDir, outDirName)
		os.makedirs(outDir)

		return outDir, new_id

def simpleNewDir(parentOutDir, name=None):
	""" Creates a new directory with a number that increments, a date, and a name.
	e.g. if folders "001_blah" and "002_foo" exist (in parentOutDir) then this function will create "003_<date>_<name>"
	"""

	OUT_DIR_STR = '{num:03d}_{txt:s}'
	OUT_DIR_PAT = re.compile('^\d\d\d_.*')

	# find directories starting with 3 digits
	dirs = [int(f[:3]) for f in os.listdir(parentOutDir) if os.path.isdir(os.path.join(parentOutDir, f)) and OUT_DIR_PAT.match(f) is not None]
	
	if (len(dirs) == 0):
		new_id = 0
	else:
		old_id = max(dirs)
		new_id = old_id+1

	now = str(datetime.datetime.now()).replace(':','.').replace(' ', '_')

	if (name is not None):
		newname = '{date}_{name}'.format(date=now.split(' ')[0], name=name)
	else:
		newname = now

	outDir = os.path.join(parentOutDir, OUT_DIR_STR.format(num=new_id, txt=newname))
	os.makedirs(outDir)

	return outDir