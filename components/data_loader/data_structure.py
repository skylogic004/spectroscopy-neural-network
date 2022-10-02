import logging
import pandas as pd
import numpy as np
from collections import OrderedDict

class CurrentData():
	""" Holds data for the current fold or bootstrap (train, dev, and test sets for current training run) """

	def __init__(self, data_dict_fold=None, use_unsupervised=False):
		self.data_dict_fold_filtered = OrderedDict()
		self.sample_primary_key = 40000 # sample IDs will start from this number

		if (data_dict_fold is not None):
			self.data_dict_fold = data_dict_fold

			if (not use_unsupervised and 'unsuper' in data_dict_fold):
				# remove unsupervised data
				self.data_dict_fold.pop('unsuper')

			# update index of all DataFrames with unique IDs
			for _set, df in self.data_dict_fold.items():
				self.update_index_with_unique_IDs(df)
		else:
			self.data_dict_fold = OrderedDict()

	def update_index_with_unique_IDs(self, df):
		# get last used sample ID, and start next IDs from there
		start_ID = self.sample_primary_key
		stop_ID = start_ID + df.shape[0]
		df.index = np.arange(start_ID, stop_ID)

		# update ID tracker, so that other DataFrames's indexes will have a unique sample ID
		self.sample_primary_key = stop_ID

		# rename the index, so we can identify when data is indexed in the same way,
		# and therefore can cross-reference samples by this index
		assert df.index.name is None
		df.index.name = 'CurrentDataIndex'

	def keys(self, include_unsuper=True):
		keys = list(self.data_dict_fold.keys())

		has_unsuper = 'unsuper' in keys

		if (has_unsuper):
			keys.remove('unsuper')

		if (has_unsuper and include_unsuper):
			# this appends it to the end when included; Always at the end.
			keys.append('unsuper')

		return keys

	def set_unsuper(self, df):
		self.update_index_with_unique_IDs(df)
		self.data_dict_fold['unsuper'] = df

	def _generate_filtered_data(self, _set):		
		# generate the filtered data if not already created
		if (_set not in self.data_dict_fold_filtered):
			df = self.get(_set, filtered=False, combine_train_dev=False)
			if (df is not None):
				not_ignored_mask = ~df['ignore']
				self.data_dict_fold_filtered[_set] = df.loc[not_ignored_mask]
			else:
				self.data_dict_fold_filtered[_set] = None

	def get(self, _set, filtered, combine_train_dev):
		""" Get data from data_dict_fold and do the following:
		     - if filtered requested, then filter the data using the "ignore" flag 
		     - if combine_train_dev requested, then combine data from train set and dev set
		"""

		if (not combine_train_dev):
			if (not filtered):
				# this is the typical usage (combine_train_dev=False and filtered=False)
				return_df = self.data_dict_fold[_set]
			else:
				# return filtered data for requested set
				self._generate_filtered_data(_set)
				return_df = self.data_dict_fold_filtered[_set]
		else:
			if (_set == 'dev'):
				raise(ValueError('dev set not available. Since combine_train_dev requested, dev set will be moved into train set.'))

			elif (_set == 'train'):
				# combine train and dev sets (recursively calls current function, but with combine_train_dev turned off)
				train_df = self.get('train', filtered, combine_train_dev=False)
				dev_df = self.get('dev', filtered, combine_train_dev=False)

				return_df = pd.concat([train_df, dev_df], axis=0)

				# sanity check
				assert return_df.shape[1] == train_df.shape[1] == dev_df.shape[1]
				assert return_df.shape[0] == train_df.shape[0] + dev_df.shape[0]

			elif (_set == 'test'):
				# combine_train_dev isn't relevant to test set, so just call this function again
				# but with that flag turned off.
				return_df = self.get('test', filtered, combine_train_dev=False)

		if (return_df is not None and return_df.shape[0] > 0):
			return return_df
		else:
			return None
