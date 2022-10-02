"""
This sets up the hyperparameter space for hyperopt to search over; 
a sample of this space is converted into a "cmd" (i.e. the command line arguments for my python script).
`HPO_start_master.py` calls `get_cmd_space`
"""

from math import ceil
from hyperopt import hp
from pprint import pprint, pformat
from hyperopt.pyll.stochastic import sample
import numpy as np
from hyperopt.pyll.base import scope
from scipy.stats import norm
import math

__author__ = "Matthew Dirks"

def get_cmd_space(which_cmd_space):
	""" This function declares the space of hyperparameters to optimize over.
	Since these parameters are ultimately command-line arguments, I call it "cmd_space".
	"""
	from components.hyperopt_config.cmd_space_for_paper import CMD_SPACE_SETTINGS
	SPACE_NAMES = list(CMD_SPACE_SETTINGS.keys())

	if (which_cmd_space in SPACE_NAMES):
		from components.hyperopt_config import cmd_space_for_paper
		cmd_space, hyperhyperparams = cmd_space_for_paper.get_cmd_space(which_cmd_space)
	else:
		raise(ValueError('Invalid cmd_space name.'))

	return cmd_space, hyperhyperparams

#=====================================================================================


def my_pformat(a_dict, width=200):
	""" pformat but hide some things for readability.
	`width` of 200 is good for printing on-screen in Jupyter notebook,
	but more like 500 is good for saving to file.
	"""

	is_iter = lambda x: isinstance(x, tuple) or isinstance(x, list)

	def _helper(d):
		if (isinstance(d, dict)):
			for key, value in d.items():
				if (key == 'kernel_init' and is_iter(value)):
					d[key] = '[HIDDEN_FOR_READABILITY]'
				elif (isinstance(value, dict)):
					d[key] = _helper(value)
				elif (is_iter(value)): # is it iterable?
					value = list(value) # make sure it's mutatable
					for idx, entry in enumerate(value):
						value[idx] = _helper(entry)

		return d

	new_dict = _helper(a_dict)
	return pformat(new_dict, width=width)

def my_pprint(a_dict, width=200):
	print(my_pformat(a_dict, width))


def print_a_sample(which_cmd_space):
	""" Just for testing/debugging: make sure we can sample from the cmd_space and check that contents are reasonable """
	cmd_space = get_cmd_space(which_cmd_space)

	print('==== CMD_SPACE ====')
	# this works ok, but has long lines
	# print(pformat(cmd_space, width=2000))

	# better: hide some details
	my_pprint(cmd_space)


	print('==== SAMPLE (NOT YET PROCESSED) ====')
	a_sample = sample(cmd_space)
	my_pprint(a_sample)

	print('==== AFTER PROCESSING ====')
	my_pprint(process_sample(a_sample))

if __name__ == '__main__':
	import fire
	fire.Fire(print_a_sample)