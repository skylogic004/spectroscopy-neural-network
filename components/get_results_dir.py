from os.path import exists

import components.colorama_helpers as _c

def get_results_dir(configData, resultsDir=None):
	""" get resultsDir, either from command line or config file """
	if (resultsDir is not None and exists(resultsDir)):
		pass # great
	else:
		# check config file
		assert 'paths' in configData
		assert 'results_dir' in configData['paths']
		if (exists(configData['paths']['results_dir'])):
			resultsDir = configData['paths']['results_dir']
		else:
			print(_c.red('Could not find a resultsDir that exists in either arguments or configData["paths"]["results_dir"]'))
			print(configData)
			exit(1)

	return resultsDir