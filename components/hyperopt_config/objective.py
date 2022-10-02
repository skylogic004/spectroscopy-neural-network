"""
This takes a sample of the hyperparameter space (which I call a "cmd_space") and evaluates it. 
`start_master.py` registers `calc_loss` as the function for minimization.
"""
from subprocess import Popen, PIPE, STDOUT
import uuid
import os
import pandas as pd
from os.path import join, exists
from hyperopt import STATUS_OK, STATUS_FAIL
from shutil import copyfile
import numpy as np
import socket
from sklearn.metrics import mean_squared_error
from timeit import default_timer as timer
import signal
import time
from os import listdir

from components.hyperopt_config.make_cmds_v3 import build_cmd
from components.hyperopt_config.helpers import process_sample

__author__ = "Matthew Dirks"

IS_LINUX = os.name == 'posix'
WINDOWS_MAX_LENGTH = 8190
COMPUTER_NAME = socket.gethostname()

REQUIRED_NUM_ENSEMBLE_RUNS = 15

SLEEP_TIME = 300

calc_RMSE = lambda arr1, arr2: np.sqrt(mean_squared_error(arr1, arr2))

RESULTS_FNAME = 'results.csv'
BEST_MODEL_PRED_FNAME = 'best_model_predictions.pkl'

def calc_loss(sample_of_cmd_space, hyperhyperparams):
	print('=============== STARTING objective.calc_loss =============== ')
	st = timer()

	# prepare dictionary to output the results
	return_dict = {
		'SLURM_JOB_ID': os.environ.get('SLURM_JOB_ID'),
		'status': STATUS_OK, # this is overwritten if something goes wrong
	}

	# decide where results dir should be saved
	results_dir = None
	results_dir_name = 'NOT SET YET'


	SLURM_TMPDIR = os.environ.get('SLURM_TMPDIR') # set on any SLURM HPC cluster
	SCRATCH_PATH = os.environ.get('SCRATCH') # set on ComputeCanada cluster
	if (SLURM_TMPDIR is not None):
		results_dir = SLURM_TMPDIR
		results_dir_name = 'SLURM_TMPDIR'
	else:
		results_dir = hyperhyperparams['LOCAL_RESULTS_DIR']
		results_dir_name = 'LOCAL_RESULTS_DIR from cmd_space hyperhyperparams'

	assert exists(results_dir), f'results_dir ({results_dir}) doesnt exist'

	
	print('objective.py: CWD={}'.format(os.getcwd()))

	######## DO THE WORK TO COMPUTE RESULTS FOR THIS TRIAL
	UUID = uuid.uuid4().hex
	return_dict, task_dirs = do_the_work(sample_of_cmd_space, hyperhyperparams, UUID, results_dir, return_dict)

	######## DONE PROCESSING
	# record time taken
	tt = timer()-st
	return_dict['time_duration_seconds'] = tt
	if (tt > 3600):
		print(f'time duration = {tt:0.0f} = {tt/3600:0.2f} hours')
	else:
		print(f'time duration = {tt:0.0f} = {tt/60:0.2f} minutes')

	# now that processing is done, lets copy files back to SCRATCH for storage
	if (results_dir_name == 'SLURM_TMPDIR' and SCRATCH_PATH is not None):
		copy_logs_to_scratch(task_dirs, join(SCRATCH_PATH, f'TASKDIR_{UUID}'))

	print('=============== END OF objective.calc_loss =============== ')


	return return_dict

def do_the_work(sample_of_cmd_space, hyperhyperparams, UUID, results_dir, return_dict):
	# prepare
	return_dict['UUID'] = UUID
	return_dict['outputs'] = {}
	return_dict['paths'] = {}
	return_dict['cmds'] = []
	return_dict['error_messages'] = []
	return_dict['n_gpu'] = os.environ.get('OVERRIDE_n_gpu')
	return_dict['n_in_parallel'] = os.environ.get('OVERRIDE_n_in_parallel')

	############################## settings for this hyperopt experiment (i.e. hyperhyperparams)
	num_folds = hyperhyperparams['NUM_FOLDS_FOR_HYPERTUNING']

	WHICH_TARGET = hyperhyperparams.get('WHICH_TARGET')
	WHICH_SET = hyperhyperparams.get('WHICH_SET')
	WHICH_METRIC = hyperhyperparams.get('WHICH_METRIC')

	############################# get settings that affect how we will process the results...
	DOING_ENSEMBLE = False
	DOING_ENSEMBLE |= 'n_training_runs' in sample_of_cmd_space and sample_of_cmd_space['n_training_runs'] > 1
	DOING_ENSEMBLE |= 'multi_run_strategy' in sample_of_cmd_space and sample_of_cmd_space['multi_run_strategy'] == 'ensemble'

	if (DOING_ENSEMBLE):
		WARNING_TIME_THRESHOLD = 60 # 1 minutes
	else:
		WARNING_TIME_THRESHOLD = 30 # seconds

	################################ post-process sample
	cmd_dict = process_sample(sample_of_cmd_space)
	cmd_dict['resultsDir'] = results_dir

	################################ loop over each fold
	task_dirs = [] # there will be one task dir per fold
	cmd_counter = 0
	for kth_fold in range(num_folds):
		task_dir_name = f'TASKDIR_{UUID}_cmd{cmd_counter}_k{kth_fold}'
		cmd_dict['m'] = task_dir_name
		cmd_dict['out_dir_naming'] = 'MANUAL'
		cmd_dict['kth_fold'] = kth_fold

		for key in ['n_gpu']:
			cmd_dict[key] = os.environ.get(f'OVERRIDE_{key}')
			assert cmd_dict[key] is not None
			print(f'Retrieved from environment var: {key} = {cmd_dict[key]}')

		n_in_parallel = os.environ.get('OVERRIDE_n_in_parallel', 0)
		cmd_dict['n_in_parallel'] = n_in_parallel

		cmd_ser = pd.Series(cmd_dict)
		cmd_str = build_cmd(None, cmd_ser)
		return_dict['cmds'].append(cmd_str)

		# check length (on Windows we run into issues if too long)
		if (len(cmd_str) > WINDOWS_MAX_LENGTH and not IS_LINUX):
			msg = f'cmd_str too long for windows (limit is probably {WINDOWS_MAX_LENGTH} and this length is {len(cmd_str)})'
			print(msg)
			return_dict['error_messages'].append(msg)

		# first call program to train NN (many times if doing CV)
		print(f'---------> calling cmd {cmd_str}')
		st = timer()
		p = Popen(cmd_str, shell=True, stdin=PIPE, stdout=PIPE, stderr=STDOUT, cwd='..'); cmd_counter += 1

		# save stdout (ONLY the stdout, NOT what's printed via logger.info(...) etc)
		_stdout = p.stdout.read().decode('UTF-8') # blocks until done
		training_time = timer() - st
		return_dict['outputs'][f'task_k{kth_fold}_stdout'] = '{}\n...\n{}'.format(_stdout[:800], _stdout[-500:]) # only some chars b/c if too big mongo dies
		
		# print stdout for debugging purposes (only if time was less than 10 minutes)
		if (training_time < WARNING_TIME_THRESHOLD):
			print('---------> stdout:')
			print(_stdout)
			print('---------> END OF stdout')

		# safety check: if AE took less than 10 minutes then probably something is broken.
		#               So, rather than continue failing repeatedly, lets sleep for a while
		if (training_time < WARNING_TIME_THRESHOLD and DOING_ENSEMBLE):
			print(f'WARNING: training finished suspiciously fast ({training_time} < {WARNING_TIME_THRESHOLD} s), sleeping now for {SLEEP_TIME} seconds...')
			time.sleep(SLEEP_TIME)

		task_dir_path = join(cmd_dict['resultsDir'], task_dir_name)
		return_dict['paths'][f'task_k{kth_fold}'] = task_dir_path
		task_dirs.append(task_dir_path)

		### Load consoleOutput.txt, save last N lines of text
		fp = join(task_dir_path, 'consoleOutput.txt')
		if (exists(fp)):
		    with open(fp, 'r') as f:
		        return_dict['outputs'][f'task_k{kth_fold}_log'] = ''.join(f.readlines()[-15:])

		########################## Ensembles: check that required number of models is completed ###################################################
		if (DOING_ENSEMBLE):
			fp = join(task_dir_path, BEST_MODEL_PRED_FNAME)

			if (exists(fp)):
				pkl_data = pd.read_pickle(fp)
				edf = pkl_data['ensemble_runs_df']

				assert len(pkl_data['prediction_columns'])==1, 'This code assumes there is only 1 target (output) from the NN'
				pred_target = pkl_data['prediction_columns'][0] # should be 'DM_pred'
				target = pkl_data['assay_columns'][0] # should be 'DM'
				assert target=='DM'

				# get names of the columns which have the predictions for each ensemble-model (i.e. each run)
				# e.g. "run0:DM_pred", ..., "run39:DM_pred"
				per_run_pred_columns = [x for x in edf.columns if x.endswith(pred_target)]

				for _set, set_df in edf.groupby('set', sort=False):
					# calc and store the RMSE of each training run. Useful in plotting variance in post-analysis
					RMSE_scores = []
					for col in per_run_pred_columns:
						RMSE_scores.append(calc_RMSE(set_df[target], set_df[col]))

					return_dict[f'{target}_{_set}_RMSE_per_training_run'] = RMSE_scores

				# how many models does the ensemble have?
				num_training_runs = len(per_run_pred_columns)
				
				# short-circuiting allows there to be less training runs than normal, but here we just make sure there aren't absurdly few
				if (num_training_runs < REQUIRED_NUM_ENSEMBLE_RUNS):
					return_dict = abort(return_dict, f'Ensemble collected only {num_training_runs} runs, but at least {REQUIRED_NUM_ENSEMBLE_RUNS} are required.')
					return return_dict, task_dirs
			else:
				return_dict = abort(return_dict, f'BEST_MODEL_PRED_FNAME FileNotFoundError ({fp}); Ensemble requires this.')
				return return_dict, task_dirs
		
	
	# then process the prediction results
	results_out_dir_name = f'TASKDIR_{UUID}_cmd{cmd_counter}'
	quoted_dirs = [f'"{x}"' for x in task_dirs]
	cmd = ['python', '-u', 'process_results.py', 'main'] + quoted_dirs + [
						# '--task_dir_is_parent_dir', 'False',
						'--save_dir', f'"{results_dir}"',
						# '--task_key', 'HYPEROPT_TEMP', 
						# '--interactive', 'False',
						'--out_dir_name', '"'+results_out_dir_name+'"']
	print('---------> calling process_results...', ' '.join(cmd))
	return_dict['cmds'].append(' '.join(cmd))
	p = Popen(cmd, shell=False, stdout=PIPE, stderr=PIPE, cwd='..'); cmd_counter += 1
	stdout, stderr = p.communicate()
	print('---------> stdout')
	print(stdout.decode('UTF-8').strip())
	print('---------> stderr')
	print(stderr.decode('UTF-8').strip())
	print('---------')

	results_out_dir_path = join(cmd_dict['resultsDir'], f'{results_out_dir_name}_results')
	
	########################## read in prediction score ###################################################
	fp = join(results_out_dir_path, RESULTS_FNAME)
	if (exists(fp)):
		df = pd.read_csv(fp)

		# store this whole file into database:
		return_dict[RESULTS_FNAME] = df.to_json()

		# get some of the prediction scores to store directly in hyperopt database:
		for target, group_df in df.groupby('target'):
			tmp_df = group_df.set_index('set')
			
			for _set in ['dev', 'test']:
				return_dict[f'{target}_{_set}_RMSE'] = tmp_df.loc[_set, 'RMSE']

	else:
		msg = f'RESULTS_FNAME FileNotFoundError ({fp})'
		print(msg)
		return_dict['error_messages'].append(msg)
		return_dict[RESULTS_FNAME] = 'FILE_NOT_FOUND'
	

	########################## record loss value ###################################################
	print(f'WHICH_TARGET = {WHICH_TARGET}, WHICH_SET = {WHICH_SET}, WHICH_METRIC = {WHICH_METRIC}')

	# WHICH_TARGET is the name of a target variable (e.g. "Cu" or "DM" etc)
	loss_var_name = f'{WHICH_TARGET}_{WHICH_SET}_{WHICH_METRIC}'
	if (loss_var_name in return_dict):
		return_dict['loss'] = return_dict[loss_var_name]
	else:
		return_dict = abort(return_dict, f'No loss data to record because {loss_var_name} key missing')
		return return_dict, task_dirs

	# return successful results
	return return_dict, task_dirs



def abort(return_dict, reason_message):
	""" If aborting due to error, pass in the message here.
	Returns updated return_dict
	"""
	txt = f'Aborted (and status set to STATUS_FAIL); {reason_message}'
	print(txt)
	return_dict['status'] = STATUS_FAIL
	return_dict['reason_for_failure'] = txt
	return return_dict



KEEP_THESE_FILES = ['consoleOutput.txt', 'eventsHistory.csv', 'lossHistory.csv', BEST_MODEL_PRED_FNAME, 'run_00/loss_history.png', 'result_info.json', 'cmd_args.pyon']
def copy_logs_to_scratch(task_dirs, outDir):
	if (not exists(outDir)):
		os.makedirs(outDir)

		for idx, task_dir_path in enumerate(task_dirs):
			for keep_fname in KEEP_THESE_FILES:
				fp = join(task_dir_path, keep_fname)
				if (exists(fp)):
					print(f'Copying {fp}')
					new_fname = keep_fname.replace('/', '-')
					copyfile(fp, join(outDir, f'{idx}_{new_fname}'))

			# keep any ensemble training run outputs (e.g. training_run_000.pkl)
			fnames = [fname for fname in listdir(task_dir_path) if fname.startswith('training_run_')]
			for fname in fnames:
				fp = join(task_dir_path, fname)
				print(f'Copying {fp}')
				copyfile(fp, join(outDir, f'{idx}_{fname}'))
	else:
		print(f'ERROR (copy_logs_to_scratch): outDir ({outDir}) already exists')
