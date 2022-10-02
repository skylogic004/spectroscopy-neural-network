from tabulate import tabulate
from os.path import join
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from textwrap import indent
from collections import OrderedDict
import pickle
import json

from components.spectroscopy_NN import plotters
from components.ensembling import Ensembler, rmse

def indent_and_log(txt):
	logger = logging.getLogger()
	logger.info(indent(txt, ' '*4))

def get_run_out_dir(ith_run, out_dir):
	return join(out_dir, f'run_{ith_run:02d}')

def finish_ensemble(out_dir, configData, ensembler, target_columns_in_use, kth_fold, sets_in_use):
	logger = logging.getLogger()

	assert len(target_columns_in_use)==1, 'This code written for one target only'
	target = target_columns_in_use[0]

	# process results into ensemble
	to_save = ensembler.make_final(target_columns_in_use)

	# save predictions to pkl
	logger.info('Saving best_model_predictions.pkl...')
	to_save['kth_fold'] = kth_fold
	with open(join(out_dir, 'best_model_predictions.pkl'), 'wb') as f:
		pickle.dump(to_save, f)

	# plot predictions
	if (configData['plot']['target_vs_predictions']):
		logger.info('Plotting final version of targets vs predictions (individual plots)')

		plotters.plot_target_vs_predictions_individual(
			out_dir, 
			'final', 
			to_save['df'],
			target_columns_in_use,
		)

	# save human-readable result info
	ens_df = to_save['df']
	result_info = OrderedDict()
	for _set in sets_in_use:
		ens_subset_df = ens_df[ens_df['set']==_set]
		result_info[f'ensemble_RMSE_{_set}'] = rmse(ens_subset_df[target], ens_subset_df[f'{target}_pred'])

	with open(join(out_dir, 'result_info.json'), 'w') as text_file:
		text_file.write(json.dumps(result_info, indent=0))

def finish_training_run(out_dir, configData, ith_run, run_result, sets_in_use):
	logger = logging.getLogger()
	run_out_dir = get_run_out_dir(ith_run, out_dir)

	# make output dir only if needed
	out_dir_needed = configData['plot']['loss_history']
	if (out_dir_needed):
		os.makedirs(run_out_dir)

	# print scores
	indent_and_log('Scores: ' + ', '.join([f'RMSE_{_set}={run_result[f"RMSE_{_set}"]:0.5f}' for _set in sets_in_use]))

	# print DEBUG info
	indent_and_log(f'DEBUG = \n{run_result["DEBUG"]}')

	# plot loss history
	if (configData['plot']['loss_history']):
		plotters.plot_history(join(run_out_dir, 'loss_history.png'), run_result['h.history'])

	# save these figures, if they exist
	figure_names = ['LR_finder_fig', 'LR_finder_fig_2']
	for fig_name in figure_names:
		if (fig_name in run_result):
			fig = run_result[fig_name]
			fig.savefig(join(run_out_dir, f'{fig_name}.png'))
			plt.close(fig)