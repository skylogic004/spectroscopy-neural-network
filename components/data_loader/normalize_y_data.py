import logging
import pandas as pd
import numpy as np
from collections import OrderedDict

def target_normalize(column_series, target_ranges):
	_min = target_ranges.loc[column_series.name, 'min']
	_max = target_ranges.loc[column_series.name, 'max']
	if (_max == _min): # range is 0, so just divide by max instead to make all values 1 (avoids divide-by-0 error)
		if (_max == 0):
			div = 1 # do nothing if ALL values are 0
		else:
			div = _max
	else:
		div = _max - _min
	return (column_series - _min) / div

def target_unnormalize(column_series, target_ranges):
	_min = target_ranges.loc[column_series.name, 'min']
	_max = target_ranges.loc[column_series.name, 'max']
	if (_max == _min): # avoid divide-by-0 error
		div = _max
	else:
		div = _max - _min
	return column_series * div + _min


def setup_y_data_WITHOUT_NORMALIZING(cur_data,
									 target_columns_in_use,
									 sets_in_use,
									 combine_train_dev):
	# these structures will hold data in the numpy array dtype
	Y_data_dict = OrderedDict()
	for _set in sets_in_use:
		# get data for kth fold, train/test/dev set
		# Some samples are select removed (ignored) by setting filtered=True.
		_df = cur_data.get(_set, filtered=True, combine_train_dev=combine_train_dev)

		# setup Y data
		if (_set != 'unsuper'):
			# Y_data_dict[_set] = normalize_Y_df(_df[target_columns_in_use]).values.astype(np.float32)
			Y_data_dict[_set] = _df[target_columns_in_use].values.astype(np.float32)

		if (_set == 'train'):
			# sanity check
			for col in target_columns_in_use:
				assert _df[col].isnull().sum() == 0, 'Expected no nulls, but found number of nulls >= 0 in column {}'.format(col)

	unnormalize_Y_df = lambda df: df # do nothing
	return Y_data_dict, unnormalize_Y_df

def setup_y_data_WITH_NORMALIZING(scaler_settings,
								  cur_data, 
								  target_columns_in_use, 
								  doing_supervised_network, 
								  sets_in_use, 
								  combine_train_dev,
								 ):
	""" Prepare data (normalize, filter, convert to float32) for use by tensorflow or any ML model training.
	This is considered to be "low level" compared to the CurrentData structure.
	"""

	logger = logging.getLogger()

	Y_stats = scaler_settings.get('Y_stats')

	if (Y_stats is None):
		# normalize assay (scale to between 0 and 1), because input data range is very small for some columns
		# First, get training data (we normalize to training data, NOT to test set, which means CV folds will have different normalizing constants)
		train_assay_df = cur_data.get('train', filtered=True, combine_train_dev=combine_train_dev).loc[:, target_columns_in_use]

		# sanity check
		for col in target_columns_in_use:
			assert train_assay_df[col].isnull().sum() == 0, 'Expected no nulls, but found number of nulls >= 0 in column {}'.format(col)

		target_ranges = pd.DataFrame({'min': train_assay_df.min(), 'max': train_assay_df.max()})

		logger.info('Target data (analytes aka assays) min-max normalized on training set using these min & max values: \n{}'.format(target_ranges.to_string()))
		scaler_settings['Y_stats'] = {'target_ranges': target_ranges}
	else:
		# will use stats given in argument
		logger.info('Target data (assays) min-max normalized by given stats (not computed here):\n\t{}'.format(Y_stats))
		target_ranges = Y_stats['target_ranges']

	normalize_Y_df = lambda df: df.apply(target_normalize, axis=0, args=(target_ranges,))
	unnormalize_Y_df = lambda df: df.apply(target_unnormalize, axis=0, args=(target_ranges,))

	assert normalize_Y_df(train_assay_df).isnull().sum().sum() == 0, 'Expected no nulls, but found number of nulls >= 0 in normalized data.\nNOTE: target_ranges={}'.format(target_ranges)

	# these structures will hold *normalized* data in the numpy array dtype
	Y_data_dict = OrderedDict()
	for _set in sets_in_use:
		# get data for kth fold, train/test/dev set
		# Some samples are select removed (ignored) by setting filtered=True.
		_df = cur_data.get(_set, filtered=True, combine_train_dev=combine_train_dev)

		# setup Y data
		if (_set != 'unsuper' and doing_supervised_network):
			Y_data_dict[_set] = normalize_Y_df(_df[target_columns_in_use]).values.astype(np.float32)

			# sanity-check: make sure normalizing did in fact set range to [0, 1] (on training data only)
			if (_set == 'train'):
				for column_list in Y_data_dict[_set].transpose():
					assert column_list.min() >= 0 and column_list.max() <= 1.0, 'Normalize sanity-check failed'

	return Y_data_dict, unnormalize_Y_df, scaler_settings

def setup_y_data(scaler_settings, cur_data, target_columns_in_use, doing_supervised_network, sets_in_use, combine_train_dev):
	logger = logging.getLogger()

	if (doing_supervised_network):
		if (scaler_settings['Y_type'] == 'none'):
			Y_data_dict, unnormalize_Y_df = setup_y_data_WITHOUT_NORMALIZING(cur_data, target_columns_in_use, sets_in_use, combine_train_dev)
			return Y_data_dict, unnormalize_Y_df, scaler_settings
		elif (scaler_settings['Y_type'] == 'min_max'):
			return setup_y_data_WITH_NORMALIZING(scaler_settings, cur_data, target_columns_in_use, doing_supervised_network, sets_in_use, combine_train_dev)
		else:
			raise(ValueError('Must provide a valid type of normalization to use on Y data via `scaler_settings["Y_type"]`'))
	else:
		assert 'Y_type' not in scaler_settings
		unnormalize_Y_df = lambda df: df # do nothing
		return None, unnormalize_Y_df, scaler_settings