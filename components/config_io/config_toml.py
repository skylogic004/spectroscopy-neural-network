#!python3

import os
from os.path import exists
import shutil
import toml

def getConfigOrCopyDefaultFile(configFpath, defaultFpath):
	cfgDict = None
	err = ''
	if (exists(configFpath)):
		cfgDict, err = getConfig(configFpath)
	elif (defaultFpath is not None):
		if (exists(defaultFpath)):
			# config file doesn't exist, so we will copy a default configuration file in order to create it
			shutil.copyfile(defaultFpath, configFpath)
			return getConfigOrCopyDefaultFile(configFpath, defaultFpath=None) # read the copied file
		else:
			err = 'Config file (%s) does not exist, and base (default) config file also does not exist (%s).' % (str(configFpath), str(defaultFpath))
	else:
		# config file doesn't exist, and we are not creating a new one by copying from somewhere else
		err = 'Config file (%s) does not exist.' % str(configFpath)

	return cfgDict, err


def getConfig(configFpath):
	if (not exists(configFpath)):
		errorMessage = 'config file doesn\'t exist'
		return None, errorMessage
	else:
		# read existing config file
		with open(configFpath, 'r') as f:
		    data = toml.load(f)

		return data, None