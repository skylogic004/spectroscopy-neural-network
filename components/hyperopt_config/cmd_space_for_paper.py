"""
Target dataset: Mangoes, with outliers removed by Dario
"""

from math import ceil
from hyperopt import hp
from pprint import pprint, pformat
from hyperopt.pyll.stochastic import sample
import numpy as np
from hyperopt.pyll.base import scope
from scipy.stats import norm
import math

from components.hyperopt_config.helpers import hp_ordinal_randint, hp_better_quniform, round_to_n, round_to_3, hp_idx

__author__ = "Matthew Dirks"

TRAIN_SET_SIZES = {
	'rand_split': 6642,
	'D2017_split': 5045,
	'D3_split': 6642,
	'D1_split': 6642,
	'D2_split': 6642,
}


CMD_SPACE_SETTINGS = {
	# ENSEMBLE EXPERIMENTS
	'ensembles-HPO_rand_split': {
		'LOCAL_RESULTS_DIR': 'C:/temp/mangoes_HPO/ensemble_rand_split',
		'SPLIT_NAME': 'rand_split',
		'NUM_TRAINING_RUNS': 40,
	},
	'ensembles-HPO_D2017_split': {
		'LOCAL_RESULTS_DIR': 'C:/temp/mangoes_HPO/ensemble_D2017_split',
		'SPLIT_NAME': 'D2017_split',
		'NUM_TRAINING_RUNS': 40,
	},
	'ensembles-HPO_D3_split': {
		'LOCAL_RESULTS_DIR': 'C:/temp/mangoes_HPO/ensemble_D3_split',
		'SPLIT_NAME': 'D3_split',
		'NUM_TRAINING_RUNS': 40,
	},
	'ensembles-HPO_D1_split': {
		'LOCAL_RESULTS_DIR': 'C:/temp/mangoes_HPO/ensemble_D1_split',
		'SPLIT_NAME': 'D1_split',
		'NUM_TRAINING_RUNS': 40,
	},
	'ensembles-HPO_D2_split': {
		'LOCAL_RESULTS_DIR': 'C:/temp/mangoes_HPO/ensemble_D2_split',
		'SPLIT_NAME': 'D2_split',
		'NUM_TRAINING_RUNS': 40,
	},


	# "SINGLE" (NON-ENSEMBLE) EXPERIMENTS
	'singles-HPO_rand_split': {
		'LOCAL_RESULTS_DIR': 'C:/temp/mangoes_HPO/single_rand_split',
		'SPLIT_NAME': 'rand_split',
		'NUM_TRAINING_RUNS': 1,
	},
	'singles-HPO_D2017_split': {
		'LOCAL_RESULTS_DIR': 'C:/temp/mangoes_HPO/single_D2017_split',
		'SPLIT_NAME': 'D2017_split',
		'NUM_TRAINING_RUNS': 1,
	},
	'singles-HPO_D3_split': {
		'LOCAL_RESULTS_DIR': 'C:/temp/mangoes_HPO/single_D3_split',
		'SPLIT_NAME': 'D3_split',
		'NUM_TRAINING_RUNS': 1,
	},
}

def make_odd(x):
	""" Make x odd by adding 1 if the x is even. """
	x = int(x)
	if ((x % 2) == 0):
		return x + 1
	else:
		return x

def get_cmd_space(which_cmd_space):
	cmd_space = _get_cmd_space(which_cmd_space)

	hyperhyperparams = {
		'LOCAL_RESULTS_DIR': CMD_SPACE_SETTINGS[which_cmd_space]['LOCAL_RESULTS_DIR'],
		'NUM_FOLDS_FOR_HYPERTUNING': 1,

		# hyperopt will optimize for dev set's target (DM) RMSE
		'WHICH_TARGET': 'DM',
		'WHICH_SET': 'dev',
		'WHICH_METRIC': 'RMSE',
	}

	return cmd_space, hyperhyperparams


def _get_cmd_space(which_cmd_space):
	split_name = CMD_SPACE_SETTINGS[which_cmd_space]['SPLIT_NAME']

	def func(cmd_dict):
		batch_size_idx = cmd_dict['$batch_size_idx']
		batch_size = cmd_dict['$BATCH_SIZE_OPTIONS'][batch_size_idx]

		# convert batch size of "full" to actual number
		if (batch_size == 'full'):
			batch_size = TRAIN_SET_SIZES[split_name]

		LR_idx = cmd_dict['$LR_idx']
		LR = cmd_dict['$LR_OPTIONS'][LR_idx]
		

		# select which combo of conv widths to use
		idx = cmd_dict['$conv_filter_width_idx']
		conv_filter_width = cmd_dict['$CONV_WIDTH_OPTIONS'][idx]

		### FC layers:
		# select which FC size to use
		idx = cmd_dict['$FC_size_idx']
		fc_size = cmd_dict['$FC_SIZE_OPTIONS'][idx]

		# select which FC size multiplier to use
		idx = cmd_dict['$subsequent_FC_size_multiplier_idx']
		fc_mul = cmd_dict['$FC_MULTIPLIERS'][idx]
		
		FC_layer_sizes = []
		tmp_size = fc_size
		for idx in range(cmd_dict['$n_FC_layers']):
			FC_layer_sizes.append(tmp_size)

			# adjust param values for the next layer (if any)
			tmp_size = max(1, int(tmp_size * fc_mul))

		### Save to cmd_dict
		cmd_dict.update({
			'LR': LR,
			'batch_size': batch_size,
			'conv_filter_width': conv_filter_width,
			'FC_layer_sizes': FC_layer_sizes,
		})

		# remove the key:values that were used in generating the layers_graph
		for key in list(cmd_dict.keys()):
			if (key.startswith('$')):
				del cmd_dict[key]
				
		return cmd_dict


	#### CONV LAYER WIDTHS ####
	base = 1.4
	CONV_WIDTH_OPTIONS = remove_dups([make_odd(base**i) for i in range(11)])
	CONV_WIDTH_OPTIONS.remove(1) # don't do width=1, that's just silly
	# so, options are [3, 5, 7, 11, 15, 21, 29]

	#### Amount of regularization ####
	L2_reg_scale = hp.loguniform('L2_reg_scale', np.log(1e-4), np.log(1))
				
	#### define constants to be used next code block ####
	options = {
		'$LR_OPTIONS': [0.005], # [0.040, 0.020, 0.010, 0.005],
		'$BATCH_SIZE_OPTIONS': [128], # [128, 256, 384, 512, 640, 768, 1024, 2048, 4096, 'full'],
		'$CONV_WIDTH_OPTIONS': CONV_WIDTH_OPTIONS, 

		'$FC_SIZE_OPTIONS': [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 96],
		#  ^(it's every 4th. It can be divided in half 2 times evenly which is nice when there are 3 layers) (count=24)

		'$FC_MULTIPLIERS': [0.5], #[0.5, 0.75],
	}
	cmd_space = {}
	cmd_space.update(options)
	cmd_space.update({
		'dataset_name': 'mangoes_Dario', # 'mangoes' = whole dataset, with outliers. 'mangoes_Dario' = outliers removed (and pre-processed)
		'input_features': ['NIR_preprocessed_and_outliers_removed'],

		# experiments will different splits:
		'fold_spec': {'type': split_name}, 

		'$LR_idx': 0,
		'$batch_size_idx': 0,
		'n_full_epochs': 750, 

		'LR_sched_settings': {'type': 'ReduceLROnPlateau'},

		# ENSEMBLING: 
		# for hyper-parameter tuning, I find 40 is fine (any less is still too unstable)
		'n_training_runs': CMD_SPACE_SETTINGS[which_cmd_space]['NUM_TRAINING_RUNS'],

		### Architecture hyperparams
		# NOTE! As in original paper, regularizers apply to BOTH conv kernel and FC weights using the SAME L2 scale
		'FC_L2_reg_scale': L2_reg_scale,
		'conv_L2_reg_scale': L2_reg_scale,

		'conv_n_filters': hp_better_quniform('conv_n_filters', 1, 13, 3),
		'$conv_filter_width_idx': hp_idx('conv_filter_width_idx', options['$CONV_WIDTH_OPTIONS']),
		'conv_filter_init': 'he_normal',

		'$n_FC_layers': hp_ordinal_randint('n_FC_layers', 1, 4),
		'$FC_size_idx': hp_idx('FC_size_idx', options['$FC_SIZE_OPTIONS']),
		'$subsequent_FC_size_multiplier_idx': 0, #hp_idx('subsequent_FC_size_multiplier_idx', _enc_space['$FC_MULTIPLIERS']),
		'FC_init': 'he_normal', # hp.choice('FC_init', ['xavier', 'he_normal']),

		'$process_function': func,
	})

	return cmd_space

def remove_dups(seq):
	seen = set()
	seen_add = seen.add
	return [x for x in seq if not (x in seen or seen_add(x))]
