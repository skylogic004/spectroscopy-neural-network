import pandas as pd
from collections import OrderedDict
import logging
import numpy as np
from sklearn.metrics import mean_squared_error
from os.path import join
import pickle

from components.spectroscopy_NN.compute_outputs import get_assay_predictions

__author__ = "Matthew Dirks"

rmse = lambda arr1, arr2: np.sqrt(mean_squared_error(arr1, arr2))

class Ensembler:
	def __init__(self):
		self.metadata_of_models = OrderedDict()

	def save_test_set_predictions(self, 
		                          outDir,
		                          n_cold_restarts,
		                          epoch,
		                          sess,
		                          sets_in_use_except_unsuper,
		                          assay_output,
		                          Xs,
		                          X_data_dict,
		                          y_true_assay,
		                          Y_data_dict,
		                          unnormalize_Y_df,
		                          target_columns_in_use,
		                          cur_data,):

		# This will overwrite existing predictions for the current training run, if any

		nth_training_run = n_cold_restarts # starting from 0

		joined_df = get_assay_predictions(
			sess,
			sets_in_use_except_unsuper,
			assay_output,
			Xs,
			X_data_dict,
			y_true_assay,
			Y_data_dict,
			unnormalize_Y_df,
			target_columns_in_use,
			cur_data,
		)

		save_data = {
			'predictions_df': joined_df,
			'epoch': epoch,
		}

		self.metadata_of_models[nth_training_run] = save_data

		# also, save to disk in case program explodes
		with open(join(outDir, f'training_run_{n_cold_restarts:03d}.pkl'), 'wb') as f:
			pickle.dump(save_data, f)

	def get_epochs(self):
		""" This info used to plot events on loss history plot """
		return [model_metadata['epoch'] for model_metadata in self.metadata_of_models.values()]

	def make_final(self, target_columns_in_use):
		""" compute mean predictions for each instance from the collection (ensemble)
		of models' predictions. And return data for saving and plotting. """
		logger = logging.getLogger()
		logger.info(f'Ensembler collected {len(self.metadata_of_models)} models. Computing the ensemble now...')

		# collect ensemble data
		if (len(self.metadata_of_models) == 0):
			# no models were saved! This is bad.
			raise(Exception('Ensembler has 0 models to work with. Cannot proceed.'))

		data = {}
		data['set'] = self.metadata_of_models[0]['predictions_df']['set']

		per_run_pred_columns = {target:[] for target in target_columns_in_use}

		for target in target_columns_in_use:
			# get ground truth from first model in the ensemble (because they're all the same)
			data[target] = self.metadata_of_models[0]['predictions_df'][target]

			# get predictions from each model in the ensemble
			for run_idx, metadata in self.metadata_of_models.items():
				df = metadata['predictions_df']
				new_col = f'run{run_idx}:{target}_pred'
				per_run_pred_columns[target].append(new_col)

				# copy predictions from the model's predictions dataframe
				data[new_col] = df[f'{target}_pred']

			
		# join dfs into one big one
		ensemble_runs_df = pd.DataFrame(data)


		# get stats (mean, etc)
		def stats_per_row(row):
			new_row = OrderedDict()

			for target in target_columns_in_use:
				predictions = row[per_run_pred_columns[target]]

				new_row[target] = row[target] # copy of ground truth value
				new_row[f'{target}_pred'] = predictions.mean()
				new_row[f'{target}_min'] = predictions.min()
				new_row[f'{target}_max'] = predictions.max()
				new_row[f'{target}_std'] = predictions.std()
			
			new_row['set'] = row['set']
			return pd.Series(new_row)
			
		stats_df = ensemble_runs_df.apply(stats_per_row, axis=1)


		to_save = {
			'df': stats_df,
			'assay_columns': target_columns_in_use,

			# NOTE: plotting functions hardcoded to use "TARGET_pred". If you change it here, plotting functions will need to change too
			'prediction_columns': [f'{target}_pred' for target in target_columns_in_use],

			# also save individual predictions from each model of the ensemble - for easy post-analysis
			'ensemble_runs_df': ensemble_runs_df,
		}

		return to_save


	def get_ensemble_RMSE(self, _set, target):
		""" Get the ensemble's score for _set and target
		(the ensemble is built using training runs completed thus far)
		"""

		# get predictions
		df = self.make_final([target])['df']

		# calc prediction accuracy
		set_df = df[df['set']==_set]
		RMSE = rmse(set_df[target], set_df[f'{target}_pred'])

		return RMSE

	def trigger_short_circuit(self, ensemble_short_circuit):
		""" Short-circuit training based on criteria supplied, if any.
		
		Returns:
			True if training should abort.
		"""
		if (ensemble_short_circuit is not None):
			logger = logging.getLogger()

			n_runs_completed = len(self.metadata_of_models)

			for settings_dict in ensemble_short_circuit:
				# e.g. settings_dict may be {'at_n_runs': 10, 'RMSE_needed': 0.8, 'target': 'DM', 'set': 'dev'}

				# Check if required number of runs have completed at this point
				if (settings_dict['at_n_runs'] == n_runs_completed):
					logger.debug(f'DEBUG INFO FOR SHORT CIRCUIT: at_n_runs = n_runs_completed = {n_runs_completed}')

					# Check if current ensemble RMSE meets requirement
					RMSE = self.get_ensemble_RMSE(settings_dict['set'], settings_dict['target'])
					logger.debug(f'DEBUG INFO FOR SHORT CIRCUIT: RMSE={RMSE}, RMSE_needed={settings_dict["RMSE_needed"]}')
					if (RMSE > settings_dict['RMSE_needed']):
						# requirement not satisfied - abort
						return True


		return False
