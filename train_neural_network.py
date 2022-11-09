import copy
import datetime
import matplotlib as mpl
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from os.path import join, split, realpath, dirname, splitext, isfile, exists, isdir
from os import listdir
import logging
import pickle
from collections import OrderedDict, namedtuple
from colorama import Fore, Back, Style
import colorama
import fire
from timeit import default_timer as timer
import inspect
import pprint
import uuid
import socket
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import sleep
import sys
APP_DIR = split(realpath(__file__))[0]
sys.path.append(APP_DIR)

from components.config_io.config_toml import getConfig, getConfigOrCopyDefaultFile
from components.prepare_out_dir_and_logging import prepare_out_dir_and_logging, cleanup_tmp_link
from components.data_loader import generic_loader
from components.data_loader.normalize_x_data import setup_x_data
from components.data_loader.normalize_y_data import setup_y_data
from components.get_results_dir import get_results_dir
from components.cmd_line_helper import get_main_function
from components.ensembling import Ensembler

__author__ = "Matthew Dirks"

mpl_logger = logging.getLogger('matplotlib') 
mpl_logger.setLevel(logging.WARNING)
pb_logger = logging.getLogger("pushbullet.Listener")
pb_logger.setLevel(logging.WARNING)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # suppress some tensorflow messages
#from tensorflow.python.util import deprecation as tfdeprecation
#tfdeprecation._PRINT_DEPRECATION_WARNINGS = False


UUID = uuid.uuid4().hex

IS_LINUX = os.name == 'posix'
if (IS_LINUX):
	SCRATCH_PATH = os.environ.get('SCRATCH') # set on ComputeCanada cluster
	if (SCRATCH_PATH is not None):
		SYMLINK_FPATH = join(SCRATCH_PATH, 'tmp_symlink_to_results_'+UUID)
	else:
		SYMLINK_FPATH = os.path.expanduser('~/tmp_symlink_to_results_'+UUID)
else: # is Windows OS
	SYMLINK_FPATH = None

COMPUTER_NAME = socket.gethostname()

TASK_KEY = 'AEv5'
CACHE_DIR = join(APP_DIR, 'data')
CHECKPOINTS_DIR = 'checkpoints'
BEST_CHECKPOINT_DIR = 'best_checkpoint'
CONFIG_FPATH = "config.toml"
CONFIG_DEFAULT_FPATH = "config_DEFAULT.toml"

if (IS_LINUX): # probably a server
	RECHECK_CONFIG_EVERY_N = 5000 # 5000 normally, 100 for local testing
else: # Windows - so probably my local PC
	RECHECK_CONFIG_EVERY_N = 100

SETS = ['train', 'dev', 'test']
NOT_TEST_SETS = ['train', 'dev']
SETS_EXCEPT_DEV = ['train', 'test']

use_unsupervised=False

class PretendFuture:
	""" This pretends to be a "future" from the `concurrent` module (for when parallelization not in use) """
	def __init__(self, the_result):
		self.the_result = the_result
	def result(self):
		return self.the_result

def do_training_run(ith_run, out_dir, the_data, hyperparams, configData, sets_in_use, target_columns_in_use, seed_start=None):
	""" This runs in a subprocess """

	# imports are here to make subprocesses work
	import tensorflow as tf

	gpus = tf.config.experimental.list_physical_devices('GPU')
	for gpu_instance in gpus: 
		tf.config.experimental.set_memory_growth(gpu_instance, True)
	DEBUG = f'[do_training_run] GPUs: {gpus}'


	if (seed_start is not None):
		seed = seed_start + ith_run
	else:
		seed = None

	if (hyperparams['run_baseline']):
		from components.spectroscopy_NN.baseline_model import reset_then_init_then_train
		DEBUG += '\nrun_baseline = True, doing baseline model'
	else:
		from components.spectroscopy_NN.dynamic_model import reset_then_init_then_train
		DEBUG += '\nrun_baseline = False, doing dynamic model'

	run_result = reset_then_init_then_train(seed, the_data, hyperparams, configData, sets_in_use, target_columns_in_use)
	run_result['DEBUG'] = DEBUG + '\n' + run_result['DEBUG']

	return ith_run, run_result
	
def tf_check():
	""" Verify tensorflow version and that GPU is available. This runs in a subprocess because tensorflow behaves 
	poorly if imported in a main process AND in a child process, so I ONLY import it in child processes
	"""
	print('\n' + '='*100 + '\n')
	try:
		import tensorflow as tf
		if (COMPUTER_NAME == 'ainaz-PC'):
			assert tf.__version__ == '2.7.0' # could only manage to get 2.7 installed... may need to change it?
		else:
			assert tf.__version__ == '2.6.0'
	except:
		raise(ImportError('tensorflow not found; be sure to use correct environment (conda activate tf2)'))

	gpus = tf.config.experimental.list_physical_devices('GPU')
	print('[tf_check] GPUs: ', gpus)

	print('\n' + '='*100 + '\n')
	return len(gpus)


def path_for_tf(fpath, out_dir):
	# tensorflow is sensitive to special characters,
	# so we use a symlink to the output directory so TF doesnt see the special characters.
	# This function swaps the original output directory with the symlink
	if (IS_LINUX):
		return fpath.replace(out_dir, SYMLINK_FPATH)
	else:
		return fpath

def build_and_train(resultsDir=None,
					m=None,
					out_dir_naming='AUTO', # alternative: MANUAL which uses `m` as output directory name
					cmd_args_fpath=None,
					dataset_name='mangoes_Dario',
					input_features=['NIR_preprocessed_and_outliers_removed'],
					fold_spec={'type':'rand_split'},
					n_gpu=1,
					n_in_parallel=None,
					seed_start=None,
					kth_fold=0,

					# hyperparams:
					n_training_runs=None,
					LR=None,
					n_full_epochs=None,
					batch_size=None,

					# hyperparams (for architecture)
					conv_filter_init=None,
					conv_filter_width=None,
					conv_L2_reg_scale=None,
					conv_n_filters=None,
					FC_init=None,
					FC_L2_reg_scale=None,
					FC_layer_sizes=None,

					run_baseline=False,
					do_ES=True,

					which_targets_set='ALL',

					LR_sched_settings=None,
					):

	# imports are here to make subprocesses work
	from components.spectroscopy_NN.end_of_training import finish_training_run, finish_ensemble
	
	if (resultsDir is None and COMPUTER_NAME == 'MD-X'):
		resultsDir = 'C:/temp/spectra_ai_results/@2020-10-08_MoreDatasets/2022-04-19_Mangoes_TF2'

	assert n_in_parallel is not None, 'Please specify'
	assert n_training_runs is not None, 'Please specify'

	# bash sometimes pases 0-padded numbers which get intrepreted as a string, rather than int (e.g. "01")
	if (seed_start is not None):
		seed_start = int(seed_start)

	# these get passed-through to each "worker" that will each do one training run
	hyperparams = {
		'LR': LR,
		'n_epochs': n_full_epochs,
		'batch_size': batch_size,

		'conv_filter_init': conv_filter_init,
		'conv_filter_width': conv_filter_width,
		'conv_L2_reg_scale': conv_L2_reg_scale,
		'conv_n_filters': conv_n_filters,
		'FC_init': FC_init,
		'FC_L2_reg_scale': FC_L2_reg_scale,
		'FC_layer_sizes': FC_layer_sizes,

		'run_baseline': run_baseline,

		'do_ES': do_ES, # if False, just run training through to the last `n_epochs` without early stopping
		'LR_sched_settings': LR_sched_settings,
	}

	if (run_baseline):
		assert conv_n_filters is None
		assert FC_layer_sizes is None
	else:
		for key, value in hyperparams.items():
			if (key == 'run_baseline'):
				continue
			else:
				assert value is not None, f'Please specify {key} argument'

	############################# SETTING UP ###############################
	start_datetime = datetime.datetime.now() # for printing total time in nice human-readable format
	start_time = timer() # for getting total time in seconds
	prev_time = timer()

	# dump parameters
	f_args, _, _, f_values = inspect.getargvalues(inspect.currentframe())
	cmd_args_dict = {k:v for k, v in f_values.items() if k in f_args}
	print('Args: {}'.format(pprint.pformat(cmd_args_dict, width=2000)))

	# get configuration settings
	configData, _ = getConfigOrCopyDefaultFile(CONFIG_FPATH, CONFIG_DEFAULT_FPATH)
	resultsDir = get_results_dir(configData, resultsDir)


	################### create output sub-directory within results directory ############################
	global_id_fpath = None

	if (out_dir_naming == 'AUTO'):
		if ('global_id_fpath' in configData['paths']):
			global_id_fpath = configData['paths']['global_id_fpath']

		use_task_run_id = True

	elif (out_dir_naming == 'MANUAL'):
		use_task_run_id = False

	else:
		raise(ValueError('Invalid out_dir_naming'))


	out_dir, task_run_id, task_key = prepare_out_dir_and_logging(
		TASK_KEY, 
		SYMLINK_FPATH, 
		resultsDir=resultsDir, 
		comment_message=m,
		cmd_args_dict=cmd_args_dict,
		global_id_fpath=global_id_fpath,
		fallback_resultsDir=configData['paths']['results_dir'],
		use_task_run_id=use_task_run_id,
	)
	logger = logging.getLogger()

	############################# TENSORFLOW CHECK ###############################
	logger.info('\n===TENSORFLOW CHECK===')

	logger.info(f'Existing CUDA_VISIBLE_DEVICES = {os.environ.get("CUDA_VISIBLE_DEVICES", "NOT SET")}')
	if (n_gpu == 0):
		os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

	tf_checker = ProcessPoolExecutor(1).submit(tf_check)
	n_available_gpus = tf_checker.result(60)
	if (n_gpu > 0):
		assert n_gpu <= n_available_gpus

	logger.info(f'Now CUDA_VISIBLE_DEVICES = {os.environ.get("CUDA_VISIBLE_DEVICES", "NOT SET")}')

	############################# PRINT INFO ###############################

	logger.info('\n===DEBUG INFO===')
	logger.info(f'COMPUTER_NAME: {COMPUTER_NAME}')
	#logger.info('tensorflow version: {}'.format(tf.__version__))
	logger.info('task_run_id: {}'.format(task_run_id))
	logger.info('out_dir: {}'.format(out_dir))
	logger.info('SYMLINK_FPATH: {}'.format(SYMLINK_FPATH))
	# logger.info('git revision: '+tr.get_git_revision_hash())
	logger.info('cwd: {}'.format(os.getcwd()))

	logger.info('Environment variable SLURM_ARRAY_TASK_ID: {}'.format(os.environ.get('SLURM_ARRAY_TASK_ID')))
	logger.info('Environment variable SLURM_TMPDIR: {}'.format(os.environ.get('SLURM_TMPDIR')))

	logger.info('\n===SETTING===')
	logger.info('configData = ' + str(configData))



	############################ LOAD DATA ####################################
	logger.info('===LOAD DATA===')
		
	if (dataset_name in generic_loader.EXPECTED_DATASET_NAMES):
		# load
		if (use_unsupervised):
			assert 'n_unsuper' in fold_spec and fold_spec['n_unsuper'] > 0, 'Cannot do use_unsupervised=True when there\'s no unsupervised data (because n_unsuper is 0)!'

		return_dict = generic_loader.load(dataset_name, CACHE_DIR, kth_fold, fold_spec, use_unsupervised)
	else:
		raise(ValueError('Invalid dataset_name'))

	cur_data = return_dict['cur_data']
	assay_columns_dict = return_dict['assay_columns_dict']
	feature_columns = return_dict['feature_columns'] # OrderedDict, where keys are input features and values are lists of column names



	combine_train_dev = fold_spec.get('combine_train_dev', False) # False if not specified
	logger.info(f'combine_train_dev = {combine_train_dev}')

	# setup which sets to use (train/dev/test)
	if (combine_train_dev):
		sets_in_use = SETS_EXCEPT_DEV
	else:
		sets_in_use = SETS

	sets_in_use_except_unsuper = sets_in_use.copy()
	if (use_unsupervised):
		# include unsupervised data, if available
		assert 'unsuper' in cur_data.keys()
		sets_in_use += ['unsuper']

	# setup which_targets to target
	if (which_targets_set == 'ALL'):
		which_targets = [el for el in assay_columns_dict.keys()]
	else: # specified ONE target
		which_targets = [which_targets_set]

	# get column name for each target (sometimes these are different, sometimes they are the same)
	target_columns_in_use = [assay_columns_dict[target] for target in which_targets]
	logger.info('which_targets (n={}) = {}'.format(len(which_targets), which_targets))
	logger.info('target_columns_in_use (n={}) = {}'.format(len(target_columns_in_use), target_columns_in_use))

	test_set_in_use = 'test' in sets_in_use
	doing_supervised_network = True

	### importance scales
	importance_scales = None
	if (importance_scales is None):
		importance_scales = [1] * len(input_features)
	assert len(importance_scales)==len(input_features)

	scaler_settings = {'X_type': 'none', 'Y_type': 'none'}

	X_data_dict, scaler_settings = setup_x_data(scaler_settings,
												cur_data, 
												sets_in_use, 
												input_features, 
												feature_columns, 
												importance_scales,
												combine_train_dev)

	Y_data_dict, unnormalize_Y_df, scaler_settings = setup_y_data(scaler_settings, 
																  cur_data, 
																  target_columns_in_use, 
																  doing_supervised_network, 
																  sets_in_use, 
																  combine_train_dev)


	# === Prepare data for training (from my usual format into format used by `reset_then_init_then_train(...)`)
	assert len(input_features)==1
	input_feature = input_features[0]
	the_data = {}
	for _set in sets_in_use:
		_set2 = _set.replace('train', 'cal').replace('dev', 'val')
		the_data[f'_x_{_set2}'] = X_data_dict[(_set, input_feature)]
		the_data[f'_y_{_set2}'] = Y_data_dict[_set]

	# PARALLELIZATION - n_in_parallel:
	as_completed_FUNCTION = as_completed
	futures = [] # results stored here
	#   - 0 means sequential, no process pool. 
	#   - setting to 1 is also sequential but still uses the subprocess function (for testing subprocess functionality)
	if (n_in_parallel == 0): 
		as_completed_FUNCTION = lambda _futures: _futures # override `as_completed` from `concurrent.futures`

		logger.info('\n===RUNNING SEQUENTIALLY WITHOUT SUBPROCESS===')
		for ith_run in range(n_training_runs):
			print(f'Running {ith_run}')
			result = do_training_run(ith_run, out_dir, the_data, hyperparams, configData, sets_in_use, target_columns_in_use, seed_start)
			futures.append(PretendFuture(the_result=result)) # this object pretends to be a "future" like from the ProcessPoolExecutor

	else:
		# === Schedule training runs in pool
		logger.info('\n===MAKING PROCESS POOL===')
		executor = ProcessPoolExecutor(n_in_parallel)
		for ith_run in range(n_training_runs):
			""" methods of a `future`:
				  future.done(): True or False
				  future.result(): waits for it to complete then returns result
			"""
			#the_data = deepcopy(the_data) # if there are any run-specific settings, deepcopy will be required otherwise jobs will share the same reference to the dictionary


			future = executor.submit(do_training_run, ith_run, out_dir, the_data, hyperparams, configData, sets_in_use, target_columns_in_use, seed_start)
			print(f'Submitted {ith_run}')
			futures.append(future)

	# === Wait for results to come in, and process them in the order they happen to finish
	ensembler = Ensembler()
	for future in as_completed_FUNCTION(futures):
		ith_run, run_result = future.result()
		end_datetime = datetime.datetime.now()
		logger.info(f'=== TRAINING RUN {ith_run} DONE ===')
		logger.info(f'    Time spent thus far {end_datetime - start_datetime} (hh:mm:ss.micro)')

		finish_training_run(out_dir, configData, ith_run, run_result, sets_in_use)

		# add results to ensembler
		ensembler.metadata_of_models[ith_run] = run_result

	logger.info('\n\n===ALL TRAINING RUNS COMPLETED===')

	finish_ensemble(out_dir, configData, ensembler, target_columns_in_use, kth_fold, sets_in_use)

	# success sound
	if (COMPUTER_NAME == 'MD-X'):
		try:
			import beepy
			beepy.beep('ready')
			print('(played notification sound)')
		except:
			pass

	# remove symlink, if needed
	cleanup_tmp_link(SYMLINK_FPATH)

	# done
	end_datetime = datetime.datetime.now()
	logger.info('Program finished! (started at {}, ended at {}, time spent: {} (hh:mm:ss.micro)'.format(start_datetime, end_datetime, end_datetime - start_datetime))
	return {'outDir': out_dir}

if __name__ == '__main__':
	colorama.init(wrap=False)

	fire.Fire({
		'main': get_main_function(build_and_train),
	})