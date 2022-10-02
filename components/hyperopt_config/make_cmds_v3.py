"""
Here, an 'experiment' is a set of many training runs, which I call "tasks".
"""

import fire
import pandas as pd
from collections import OrderedDict
import itertools
import numpy as np
import os
from os.path import exists, isdir
from posixpath import join, normpath
import json
import colorama
from colorama import Fore, Back, Style
import types
import pprint
import json

__author__ = "Matthew Dirks"


def dict_to_cmd_arg(ob):
	# format as JSON, no newlines and no indentation
	arg_str = json.dumps(ob, indent=None)

	# windows command line requires args to be quoted in double quotes, and therefore the inner quotes need to single quotes
	arg_str = arg_str.replace('\"', '\'')

	# wrap in double quotes
	arg_str = '"{}"'.format(arg_str)

	# the keyword 'None' is a python literal not a string (json wraps it in quotes)
	arg_str = arg_str.replace('\'None\'', 'None')

	# json converts None into 'null' (None literal to string 'null') (note: the above line may be obsolete, b/c None's don't seem to pass through)
	# convert back to None without quotes (python literal)
	arg_str = arg_str.replace('null', 'None')

	# json uses true/false as literals, but I want the Python literals for booleans
	arg_str = arg_str.replace('true', 'True')
	arg_str = arg_str.replace('false', 'False')

	return arg_str

def build_cmd(experiment_name, row):
	if ('m' not in row): # message *not* manually specified
		message = 'cmd{}'.format(row['cmd_num'])

		if ('APPEND_EXPERIMENT_NAME_TO_MESSAGE' not in row or row['APPEND_EXPERIMENT_NAME_TO_MESSAGE']):
			message += '_{}'.format(experiment_name)

		if ('APPEND_KTH_FOLD_TO_MESSAGE' in row and row['APPEND_KTH_FOLD_TO_MESSAGE'] and 'kth_fold' in row):
			message += '_k={}'.format(row['kth_fold'])
	else:
		# override: just use the given message in `m`
		message = row['m']

	assert 'model_name' not in row

	if ('n_training_runs' in row):
		python_program = 'train_neural_network.py main'

	cmd = 'python {python_program}'.format(python_program=python_program)

	cmd += ' --m "{}"'.format(message)

	# special case: if function provided, apply it to the row first
	if ('_function' in row):
		f = row['_function']['f']
		arg_names = row['_function']['arg_names']
		output_name = row['_function']['output_name']

		# call function with arguments, store result into current row
		args = [row[arg_name] for arg_name in arg_names]
		output = f(*args)
		row[output_name] = output

	# typical usage: all key-value-pairs in row are added as keyword arguments to the command line
	# by looping through them all here (and applying some conversions as needed)
	for key in row.index:
		try:
			value = row[key]

			# special case: when value is a lambda function, call it on the current row 
			# and replace current value with the returned value
			# (note: this is recursive! Lambdas that return more lambdas are allowed; they are called in succession)
			while (isinstance(value, types.LambdaType) or isinstance(value, types.FunctionType)):
				value = value(row)

			### Dealing with data types ###
			if (value is None):
				arg_value = 'None'

			# explicit surrounding quotes (strings, or arrays of primitives)
			elif (key in ['cmd_args_fpath', 'sampleForm', 'resultsDir', 'importance_scales']):
				arg_value = '"{}"'.format(value)

			# whatever values (string, float, bool, int, None)
			elif (key in ['model_name', 'LR', 'n_epochs_until_anneal', 'enc_arch', 'dec_arch', 'net_arch', 
			              'use_unsupervised', 'fraction_samples', 'which_elements_set', 'build_only_then_exit',
			              'recon_loss_weight', 'ES_generalization_error_threshold', 'init_keep_prob', 'interactive',
			              'out_dir_naming', 'min_loss_improvement', 'n_epochs_warm_restart_patience', 'n_total_warm_restarts',
			              'multi_run_strategy', 'training_run_strategy', 'n_epochs_plateau_patience', 'n_epochs_worsening_patience',
			              'training_run_max_epochs', 'shuffle_batch', 'timelimit_hours', 
			              'FC_L2_reg_scale', 'conv_filter_init', 'conv_L2_reg_scale', 'FC_init', 'run_baseline',
			              'do_LR_finder', 'do_ES', 'dataset_name']):
				arg_value = '{}'.format(value)

			# float values
			elif (key in []):
				arg_value = '{:f}'.format(value)

			# integer values
			elif (key in ['kth_fold', 'n_dropout_epochs', 'n_patience_epochs', 'n_epochs', 'n_epochs_cold_restart_patience', 
			              'n_epochs_cold_restart_plateau_patience', 'n_total_cold_restarts', 'n_gpu', 'n_cpu', 'batch_size',
			              'n_full_epochs', 'n_in_parallel', 'seed_start', 'n_training_runs', 'conv_filter_width', 'conv_n_filters']):
				arg_value = '{:d}'.format(int(value))

			# array values, where each array contains any python literal (dict, tuple, str, etc)
			elif (key in ['input_features', 'ensemble_short_circuit', 'FC_layer_sizes', 'drop_LR_at_epochs']):
				arg_value = '"{}"'.format(str(value))

			# array values, where each array contains only strings
			elif (key in ['ignore_samples']):
				array_as_string = ','.join(['\'{}\''.format(x) for x in value])
				arg_value = '"[{}]"'.format(array_as_string)

			# dict values
			elif (key in ['fold_spec', 'model_params', 'enc_options', 'dec_options', 'NN_options', 'superNN_options', 
			              'batchNames', 'LR_sched_settings', 'loss_func', 'scaler_settings']):
				arg_value = '{:s}'.format(dict_to_cmd_arg(value))

			# array of ints
			elif (key in ['kth_folds']):
				assert isinstance(value, list), '{} must be a list'.format(key)
				arg_value = '[{}]'.format(','.join([str(x) for x in value]))

			# ignore these
			elif (key in ['EXPERIMENT_NAME', 'cmd_num', 'APPEND_KTH_FOLD_TO_MESSAGE', 'APPEND_EXPERIMENT_NAME_TO_MESSAGE', 
			              'MAKE_EXPERIMENT_SUBDIR', 'RUN_COMPARE_SCRIPT', 'RUN_CVRESULTS_SCRIPT', 'SETTINGS_FOR_SBATCH',
			              'START_CMD_AT', 'ALLOW_OVERWRITE', 'TASK_KEY', 'RUN_SHIFT_PLOT_SCRIPT',
			              'm', # 'm' is handled separately above, so ignore it here.
			             ]):
				arg_value = None
				
			# ignore these
			elif (key.startswith('_')):
				arg_value = None

			else:
				raise(ValueError('make_cmd: Couldn\'t figure out what to do with key ({})'.format(key)))
		except Exception as e:
			print(f'DEBUGGING INFO: key={key}, value={value}, value is None? {value is None}, value type={type(value)}')
			raise(e)

		# append key and converted value (arg_value) to cmd string
		if (arg_value is not None):
			cmd += ' --{key} {arg_value:s}'.format(key=key, arg_value=arg_value)

	return cmd
