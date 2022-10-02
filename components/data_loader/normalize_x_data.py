import logging
import pandas as pd
import numpy as np
from collections import OrderedDict

def _get_x_data_as_ndarray_float32(cur_data, 
				   sets_in_use, 
				   input_features, 
				   feature_columns, 
				   combine_train_dev,
				   ):
	""" Grab the X data, convert to numpy array, cast as float32, and convert to 2D matrix (not vector) if needed. """
	logger = logging.getLogger()

	X_data_dict = OrderedDict()
	for _set in sets_in_use:
		# get data for kth fold, train/test/dev set
		# Some samples are select removed (ignored) by setting filtered=True.
		_df = cur_data.get(_set, filtered=True, combine_train_dev=combine_train_dev)

		# setup X data (with all input_features)
		for feature in input_features:
			# from _df, select input features which belong to sensor type(s) like XRF or HYPER
			# and convert to numpy matrix of floats, for tensorflow
			if (_df is not None):
				X_data_dict[(_set, feature)] = _df[feature_columns[feature]].values.astype(np.float32)
				logger.info('Number of samples in ({}, {}) set: {}'.format(_set, feature, _df.shape[0]))
			else:
				logger.info('Number of samples in ({}, {}) set: None (disabled)'.format(_set, feature))

			# ensure X is a matrix not a vector
			if (len(X_data_dict[(_set, feature)].shape) == 1):
				X_data_dict[(_set, feature)] = X_data_dict[(_set, feature)].reshape(-1, 1)

	return X_data_dict

def _scale_x_via_max(scaler_settings,
					 X_data_dict, 
					 sets_in_use, 
					 input_features, 
					 importance_scales):
	""" Scales each input_feature by dividing by max. 
		Does NOT move bottom to 0 because we assume that spectra start pretty close to 0 anyways.
	"""
	logger = logging.getLogger()

	# for overriding the ones that would otherwise be computed in this function
	# (stats used to be called "input_maxes")
	X_stats = scaler_settings.get('X_stats')

	# get normalizing constant for all spectra, 1 per sensor type. training (& unsuper) set only
	if (X_stats is None): # the default; compute them here
		X_stats = {}
		for feature in input_features:
			_counts_max = X_data_dict[('train', feature)].max()

			# also consider unsupervised data too, if available
			if (('unsuper', feature) in X_data_dict):
				_counts_max = max(_counts_max, X_data_dict[('unsuper', feature)].max())

			# not normalizing within model anymore. It screws with the loss function.
			# _max_tf = tf.constant(_counts_max, dtype=tf.float32)
			
			X_stats[feature] = {'max': _counts_max}
			logger.info('Input data for feature {} normalized via "max" by training (& unsupervised) set max: {:0.5f}'.format(feature, _counts_max))
		scaler_settings['X_stats'] = X_stats
	else:
		# will use stats given in argument
		logger.info('Input data will be normalized by the following weights (not computed here):\n\t{}'.format(X_stats))

	# apply normalization to all matrices
	for feature in input_features:
		for _set in sets_in_use:
			X_data_dict[(_set, feature)] /= X_stats[feature]['max']

	# apply scaling to adjust a feature to be more "important" than another
	assert len(input_features)==len(importance_scales)
	for feature, scale in zip(input_features, importance_scales):
		for _set in sets_in_use:
			X_data_dict[(_set, feature)] *= scale

	return X_data_dict, scaler_settings

def _scale_x_via_min_max(scaler_settings,
						 X_data_dict, 
						 sets_in_use, 
						 input_features):
	""" This version does normalization using min and max. Per input feature.
	"""
	logger = logging.getLogger()
	X_stats = scaler_settings.get('X_stats')


	# get normalizing constants for all spectra, 1 per sensor type. train set only.
	if (X_stats is None): # the default; compute them here
		X_stats = {}
		for feature in input_features:
			X_stats[feature] = {
				'min': X_data_dict[('train', feature)].min(),
				'max':  X_data_dict[('train', feature)].max(),
			}
			logger.info(f'Input data for feature {feature} normalized via min_max by training (& NOT unsupervised) set: {X_stats[feature]}')
		scaler_settings['X_stats'] = X_stats
	else:
		# will use stats given in argument
		logger.info('Input data will be normalized by the following weights (not computed here):\n\t{}'.format(X_stats))

	# apply normalization to all matrices (all sets)
	for feature in input_features:
		for _set in sets_in_use:
			_min = X_stats[feature]['min']
			_max = X_stats[feature]['max']

			X_data_dict[(_set, feature)] -= _min
			X_data_dict[(_set, feature)] /= _max - _min

	return X_data_dict, scaler_settings

def _scale_x_via_mean_std(scaler_settings,
						  X_data_dict, 
						  sets_in_use, 
						  input_features):
	""" This version does standardization using mean and std. Column-wise.
	"""
	logger = logging.getLogger()
	X_stats = scaler_settings.get('X_stats')

	# get stats per "column" (i.e. each band of each spectrum, where a spectrum is an "input_feature")
	# on train set only
	if (X_stats is None): # the default; compute them here
		X_stats = {}
		for feature in input_features:
			# calc mean - to subtract it
			# calc std - to divide by it
			stds = X_data_dict[('train', feature)].std(axis=0)
			num_zero_values = (stds==0).sum()
			if (num_zero_values > 0):
				logger.warning(f'In calculating std for mean_std standardization, {num_zero_values} 0 values were found. Will use 1 instead of 0 in the division.')
				stds[stds==0] = 1

			X_stats[feature] = {
				'means': X_data_dict[('train', feature)].mean(axis=0),
				'stds': stds,
			}

			logger.info(f'Input data for feature {feature} normalized via mean_std by training (& NOT unsupervised) set: {X_stats[feature]}')
		scaler_settings['X_stats'] = X_stats
	else:
		# will use stats given in argument
		logger.info('Input data will be normalized by the following weights (not computed here):\n\t{}'.format(X_stats))

	# apply normalization to all matrices (all sets)
	for feature in input_features:
		for _set in sets_in_use:
			means = X_stats[feature]['means']
			stds = X_stats[feature]['stds']

			X_data_dict[(_set, feature)] -= means
			X_data_dict[(_set, feature)] /= stds

	return X_data_dict, scaler_settings

def _scale_x_via_PowerTransformer(scaler_settings,
						          X_data_dict, 
						          sets_in_use, 
						          input_features):
	""" This version does standardization using
	https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html#sklearn.preprocessing.PowerTransformer
	Column-wise. """
	from sklearn.preprocessing import PowerTransformer

	logger = logging.getLogger()
	X_stats = scaler_settings.get('X_stats')

	# get stats per "column" (i.e. each band of each spectrum, where a spectrum is an "input_feature")
	# on train set only
	if (X_stats is None): # the default; compute them here
		X_stats = {}
		for feature in input_features:
			pt = PowerTransformer()
			pt.fit(X_data_dict[('train', feature)])

			X_stats[feature] = {
				'power_transformer': pt,
				'lambdas': pt.lambdas_,
			}
			logger.info(f'Input data for feature {feature} normalized via PowerTransformer on training (& NOT unsupervised) set: {X_stats[feature]}')
		scaler_settings['X_stats'] = X_stats
	else:
		# will use stats given in argument
		logger.info('Input data will be normalized by the following weights (not computed here):\n\t{}'.format(X_stats))
		raise(ValueError('This isnt coded yet.'))

	# apply normalization to all matrices (all sets)
	for feature in input_features:
		for _set in sets_in_use:
			pt = X_stats[feature]['power_transformer']
			X_data_dict[(_set, feature)] = pt.transform(X_data_dict[(_set, feature)])

	return X_data_dict, scaler_settings



def setup_x_data(scaler_settings,
				 cur_data, 
				 sets_in_use, 
				 input_features, 
				 feature_columns, 
				 importance_scales,
				 combine_train_dev,
				 # input_maxes=None, # for overriding the ones that would otherwise be computed in this function
				):
	""" Prepare data (normalize, filter, convert to float32) for use by tensorflow or any ML model training.
	This is considered to be "low level" compared to the CurrentData structure.
	"""

	logger = logging.getLogger()

	# get X data as numpy array with float32 dtype
	X_data_dict = _get_x_data_as_ndarray_float32(cur_data, sets_in_use, input_features, feature_columns, combine_train_dev)

	if (scaler_settings['X_type'] == 'max'):
		return _scale_x_via_max(scaler_settings, X_data_dict, sets_in_use, input_features, importance_scales)
	elif (scaler_settings['X_type'] == 'min_max'):
		assert all([val==1 for val in importance_scales]), 'importance_scales not implemented for this scaler type'
		return _scale_x_via_min_max(scaler_settings, X_data_dict, sets_in_use, input_features)
	elif (scaler_settings['X_type'] == 'mean_std'):
		assert all([val==1 for val in importance_scales]), 'importance_scales not implemented for this scaler type'
		return _scale_x_via_mean_std(scaler_settings, X_data_dict, sets_in_use, input_features)
	elif (scaler_settings['X_type'] == 'none'):
		assert all([val==1 for val in importance_scales]), 'importance_scales not implemented for this scaler type'
		return X_data_dict, scaler_settings
	elif (scaler_settings['X_type'] == 'power_transformer'):
		assert all([val==1 for val in importance_scales]), 'importance_scales not implemented for this scaler type'
		return _scale_x_via_PowerTransformer(scaler_settings, X_data_dict, sets_in_use, input_features)
	else:
		raise(ValueError('Must provide a valid type of normalization to use on X data via `scaler_settings["X_type"]`'))

