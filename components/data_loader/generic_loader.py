import pandas as pd
from sklearn.utils import shuffle
import logging
from os.path import exists, join
import pickle
from sklearn.model_selection import LeaveOneOut, LeavePOut, ShuffleSplit, KFold, RepeatedKFold
from tqdm import tqdm
from collections import OrderedDict, Counter
from sklearn.utils import resample
import numpy as np
from colorama import Fore, Back, Style

from components.data_loader.data_structure import CurrentData
from components.data_loader import mangoes

RANDOM_STATE = 1234 # seed for random sampling train/dev/test sets

DATASETS = {
	# 'pharm_tablets': pharmaceutical_tablets.load_from_source,
	# 'ashram': ashram.load_from_source,
	# 'COVID19': COVID19.load_from_source,
	'mangoes': mangoes.load_Anderson_data,
	'mangoes_Dario': mangoes.load_Dario_data,
}

EXPECTED_BATCH_NAMES = list(DATASETS.keys())


def load(dataset_name, cache_dir, kth_fold, fold_spec, use_unsupervised):
	logger = logging.getLogger()

	# get cache filename
	fname = f'tf_ready_{dataset_name}.pkl'
	fpath = join(cache_dir, fname)

	logger.info(f'Loading {fpath}')

	if (exists(fpath)):
		logger.info('Reading data from cache ({})...'.format(fpath))
		#with open(fpath, 'rb') as f:
		cache = pd.read_pickle(fpath)
	else:
		logger.info('Data cache ({}) doesn\'t exist.'.format(fpath))

		# get dataset-specific loader function
		f = DATASETS[dataset_name]

		# call loader function
		cache = f()

		with open(fpath, 'wb') as f:
			pickle.dump(cache, f)

		logger.info('Data read from source (and written to cache {}).'.format(fname))

	# data structure
	cur_data = CurrentData(get_kth_fold(kth_fold, cache['data_dict'], fold_spec), use_unsupervised=use_unsupervised)

	return_dict = cache
	return_dict['cur_data'] = cur_data

	return return_dict


def get_kth_fold(kth_fold, data_dict, fold_spec):
	"""
	Args:
		fold_spec:
			n_super: number of samples to use for supervised learning (will be further split into train, dev, test sets)
			n_unsuper: number of samples to use for unsupervised learning
			n_dev: number of samples to use for dev set; will be selected from the training set which is from the supervised set of data
			random_state: for df.sample
			r: number of CV repeats
			k: number of CV folds
	"""
	logger = logging.getLogger()

	assert all([key not in fold_spec for key in ['how_to_use_unassigned']])

	# prepare data to actually use for this particular instance of kth_fold
	# it should have these keys: unsuper, train, dev, test
	data_dict_fold = OrderedDict()

	# get data
	df = data_dict['shuffled_df']

	# ANY NAME that ends with "_split" is assumed to be a column name that labels each row with the set
	if (fold_spec['type'].endswith('_split')):
		column_with_split = fold_spec['type']

		assert all([key not in fold_spec for key in ['n_super', 'n_unsuper', 'n_dev', 'random_state', 'r', 'k']])
		assert kth_fold == 0

		# use the data splits specified in column
		data_dict_fold['unsuper'] = None
		data_dict_fold['train'] = df[df[column_with_split]=='calibrate']
		data_dict_fold['dev'] = df[df[column_with_split].isin(['validate', 'dev', 'tuning'])]
		data_dict_fold['test'] = df[df[column_with_split]=='test']

	elif (fold_spec['type'] == 'rxkCV'): # repeated k-fold cross-validation (r x k-fold CV)
		assert all([key in fold_spec for key in ['n_super', 'n_unsuper', 'n_dev', 'super_subset_random_seed', 'r', 'k']])

		n_repeats = fold_spec['r'] # number of repetitions of the whole CV procedure
		k_folds = fold_spec['k'] # number of folds per CV procedure
		assert kth_fold < n_repeats*k_folds

		n_dev = fold_spec['n_dev']
		n_super = fold_spec['n_super']
		assert n_dev < n_super, 'dev set is selected from the "super" set'
		n_unsuper = fold_spec['n_unsuper']

		# get indices from k-fold cross validation. Each repetition uses a different shuffling of the data.
		cv = RepeatedKFold(n_splits=k_folds, n_repeats=n_repeats, random_state=RANDOM_STATE)
		indices = {}
		indices['non_test'], indices['test'] = list(cv.split(df))[kth_fold]

		# test set is established here, before subsetting the rest for limited-data experiments
		data_dict_fold['test'] = df.iloc[indices['test']]
		non_test_df = df.iloc[indices['non_test']]

		# select a subset of the data to be used for supervised learning ("labelled" samples)
		logger.debug(f'non_test_df.shape={non_test_df.shape}, n_super={n_super}, num test samples={len(indices["test"])}')
		super_subset_df = non_test_df.sample(n=n_super, replace=False, random_state=fold_spec['super_subset_random_seed'])

		# remaining samples can be used for unsupervised learning (emulating "unlabelled" samples)
		unsuper_subset_df = non_test_df[~non_test_df.index.isin(super_subset_df.index)]

		# use only some of them
		if (n_unsuper == 'all'):
			data_dict_fold['unsuper'] = unsuper_subset_df
		elif (isinstance(n_unsuper, int)):
			data_dict_fold['unsuper'] = unsuper_subset_df.iloc[:n_unsuper]
		else:
			raise(ValueError('n_unsuper must be int or "all" string'))

		# split labelled ("super") data into train & dev sets
		data_dict_fold['dev'] = super_subset_df.iloc[:n_dev]
		data_dict_fold['train'] = super_subset_df.iloc[n_dev:]

	else:
		raise(ValueError('No other fold_spec type supported.'))

	return data_dict_fold
