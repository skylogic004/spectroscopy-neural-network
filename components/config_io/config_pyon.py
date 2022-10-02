""" Alternative config format.
Uses Python-parser to load config file (accepts valid python code).
"""
import ast
from os.path import exists
import shutil
# import json
import pprint

def read(fpath, copyIfDoesntExist=None):
	""" Load a PYTHON config file. """
	ob = None
	err = ''
	if (exists(fpath)):
		configFile = open(fpath, 'r')
		try:
			s = configFile.read()
			ob = ast.literal_eval(s)
		except Exception as e:
			err += str(e)

		configFile.close()
	else:
		if (copyIfDoesntExist is not None):
			# config file doesn't exist, so we will copy a default configuration file in order to create it
			if (exists(copyIfDoesntExist)):
				shutil.copyfile(copyIfDoesntExist, fpath)
				return read(fpath, copyIfDoesntExist=None) # read the copied file
			else:
				err = 'Config file (%s) does not exist, and base (default) config file also does not exist (%s).' % (str(fpath), str(copyIfDoesntExist))
		else:
			# config file doesn't exist, and we are not creating a new one by copying from somewhere else
			err = 'Config file (%s) does not exist.' % str(fpath)

	return ob, err

def write(fpath, configData):
	if (isinstance(configData, str)): # user provided exactly what they want in the config file as string
		# validate (will raise exceptions if any problems)
		ob = ast.literal_eval(configData)

		# write string to disk
		with open(fpath, 'w') as f:
			f.write(configData)
	else:
		# user provided a dictionary (recommended) or some other object,
		# now convert it and save
		with open(fpath, 'w') as f:
			txt = pprint.pformat(configData, width=1000)
			f.write(txt)