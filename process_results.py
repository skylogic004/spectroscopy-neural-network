import matplotlib as mpl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from os import listdir
from os.path import join, split, realpath, dirname, splitext, isfile, exists, isdir, getsize
import posixpath
from collections import OrderedDict
from colorama import Fore, Back, Style
import colorama
import fire
from sklearn import metrics
import math
from components.scores import score, cap_to_zero
import pickle

TASK_KEY = 'HYPEROPT_TEMP'
RESULTS_FNAME = 'results.csv'
BEST_MODEL_PRED_FNAME = 'best_model_predictions.pkl'

def load_saved_datas(predictions_fpath):
	saved_datas = None

	if (exists(predictions_fpath)):
		try:
			with open(predictions_fpath, 'rb') as f:
				ob = pickle.load(f)
					
				if (isinstance(ob, dict)): # one instance of "saved_data"
					saved_datas = [ob]
				elif (isinstance(ob, list)): # many instances of "saved_data"
					saved_datas = ob

				# cap predictions to at least 0
				for saved_data in saved_datas:
					prediction_columns = saved_data['prediction_columns']
					saved_data['capped_df'] = cap_to_zero(saved_data['df'], prediction_columns)
		except Exception as e:
			print('Debug info: predictions_fpath = {}'.format(predictions_fpath))
			raise(e)
	
	return saved_datas

def prepare_output_dir(task_key, save_dir, comment_message=None, out_dir_name=None):
	# make dir with specified name
	outDir = posixpath.join(realpath(save_dir), f'{out_dir_name}_{comment_message}')
	os.makedirs(outDir)

	return outDir

def main(task_directory, 
	     save_dir=None, 
	     out_dir_name=None):


	# load data
	print('Loading...')

	predictions_fpath = join(task_directory, BEST_MODEL_PRED_FNAME)
	saved_datas = load_saved_datas(predictions_fpath)
	if (saved_datas is None):
		raise(ValueError(f'no {BEST_MODEL_PRED_FNAME}'))
	assert len(saved_datas)==1
	saved_data = saved_datas[0]


	ASSAY_COLUMNS_KEY = 'assay_columns'
	assay_columns_in_use = saved_datas[0][ASSAY_COLUMNS_KEY]

	# map which target's predictions for which fold are where (in which saved_data)
	get_lookup_key = lambda kth_fold, target: (kth_fold, target)

	print('Detecting targets...')
	saved_datas_dict = {}
	all_targets = set()
	kth_fold = saved_data['kth_fold']
	assert kth_fold == 0, 'CV not supported here'
	all_folds = [kth_fold]

	for target in assay_columns_in_use:
		assert get_lookup_key(kth_fold, target) not in saved_datas_dict, 'saved_datas_dict already has entry for {}'.format((kth_fold, target))
		saved_datas_dict[get_lookup_key(kth_fold, target)] = saved_data
		all_targets.add(target)

	# prepare output directory
	outDir = prepare_output_dir(TASK_KEY, save_dir, comment_message='results', out_dir_name=out_dir_name)

	# save command line used to run
	with open(join(outDir, 'cmd.txt'), 'w') as text_file:
		text_file.write(' '.join(sys.argv))


	# CALCULATE SCORES FOR ALL TARGETS, FOLDS, AND SETS
	scores_table = []
	for idx, target in enumerate(all_targets):
		print('{}/{}: {}'.format(idx+1, len(all_targets), target))

		_prediction_column = target+'_pred'

		#### for saved_data in saved_datas: # per fold
		for kth_fold in all_folds:
			_df = saved_datas_dict[get_lookup_key(kth_fold, target)]['capped_df']

			# loop over: train, dev, and test sets, as well as ALL sets
			subsets = list(_df.groupby('set')) + [('ALL', _df)]
			for _set, set_df in subsets:
				MAE, MSE, RMSE, R2, num_points = score(set_df, _set, target, _prediction_column)

				row = OrderedDict()
				row['target'] = target
				row['kth_fold'] = kth_fold
				row['set'] = _set

				row['MAE'] = MAE
				row['MSE'] = MSE
				row['RMSE'] = RMSE
				row['R2'] = R2
				row['num_points'] = num_points

				scores_table.append(row)

	scores_df = pd.DataFrame(scores_table)

	scores_df.to_csv(join(outDir, RESULTS_FNAME), index=False)
	print(f'Saved scores_table {RESULTS_FNAME}')

	print('DONE: '+Fore.CYAN+outDir+Fore.RESET)

if __name__ == '__main__':
	colorama.init()
	fire.Fire({'main':main})