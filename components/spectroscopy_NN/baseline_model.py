"""
This model is designed to reproduce the one published in
	P, Mishra, D, Passos, A synergistic use of chemometrics aud deep learning improved the predictive performance of near-
	infrared spectroscopy models for dry matter prediction in mango fruit, Chemometrics and Intelligent Laboratory Systems
	212 (2021) 104287. doi:https://doi.org/10.1016/j.chemolab.2021.104287.
	URL https://www.sciencedirect.com/science/article/pii/S0169743921000551

This code is based on code by the authors of the above paper:
	https://github.com/dario-passos/DeepLearning_for_VIS-NIR_Spectra/blob/master/notebooks/Tutorial_on_DL_optimization/1)%20optimization_tutorial_regression.ipynb
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
import random
import logging
from timeit import default_timer as timer

from tensorflow.keras.callbacks import EarlyStopping, Callback, ReduceLROnPlateau
from tensorflow.keras import Sequential
from tensorflow import keras
from sklearn.metrics import mean_squared_error
from tensorflow.keras import datasets, layers, models
import pandas as pd

SETS = ['train', 'dev', 'test']

def set_all_seeds(seed):
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	random.seed(seed)
	tf.random.set_seed(seed)

def get_init_fn(init_name, seed=None):
	init_name = init_name.lower()

	if (init_name == 'unit_normal'):
		return tf.random_normal_initializer(stddev=1, seed=seed)

	elif (init_name.startswith('xavier')): 
		# xavier is another name for glorot 
		#   https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotNormal
		#   https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform
		raise(ValueError('Use glorot instead of xavier'))

	elif (init_name == 'zero'):
		return tf.initializers.constant(0)

	elif (init_name == 'glorot_normal'):
		return tf.keras.initializers.glorot_normal(seed=seed)

	elif (init_name == 'glorot_uniform'):
		return tf.keras.initializers.glorot_uniform(seed=seed)

	elif (init_name == 'he_normal'):
		return tf.keras.initializers.he_normal(seed=seed)

	elif (init_name == 'he_uniform'):
		return tf.keras.initializers.he_uniform(seed=seed)

	else:
		raise(ValueError('Invalid initialization function name: {}'.format(init_name)))
	
def init_model(hyperparams, x_train, seed=None):
	## Layers dimensions
	INPUT_DIMS = np.shape(x_train)[1]
	CONV1D_DIMS = INPUT_DIMS
	K_NUMBER = 1

	K_WIDTH = hyperparams['conv_filter_width']
	K_STRIDE = 1

	FC1_DIMS = 36
	FC2_DIMS = 18
	FC3_DIMS = 12
	OUT_DIMS = 1

	K_INIT = get_init_fn(hyperparams['conv_filter_init'], seed)
	FC_INIT = get_init_fn(hyperparams['FC_init'], seed)

	K_REG = tf.keras.regularizers.l2(hyperparams['conv_L2_reg_scale'])
	FC_REG = tf.keras.regularizers.l2(hyperparams['FC_L2_reg_scale'])

	model_cnn = keras.Sequential([  keras.layers.Reshape((INPUT_DIMS, 1),input_shape=(INPUT_DIMS,)), \
								keras.layers.Conv1D(filters=K_NUMBER, \
													kernel_size=K_WIDTH, \
													strides=K_STRIDE, \
													padding='same', \
													kernel_initializer=K_INIT,\
													kernel_regularizer=K_REG,\
													activation='elu',\
													input_shape=(CONV1D_DIMS,1)), \
								keras.layers.Flatten(),
								keras.layers.Dense(FC1_DIMS, \
												   kernel_initializer=FC_INIT, \
												   kernel_regularizer=FC_REG, \
												   activation='elu'),
								keras.layers.Dense(FC2_DIMS, \
												   kernel_initializer=FC_INIT,\
												   kernel_regularizer=FC_REG,\
												   activation='elu'),
								keras.layers.Dense(FC3_DIMS, \
												   kernel_initializer=FC_INIT, \
												   kernel_regularizer=FC_REG, \
												   activation='elu'),
								keras.layers.Dense(1, kernel_initializer=FC_INIT, \
												   kernel_regularizer=FC_REG,\
												   activation='linear'),
							  ])
	return(model_cnn)


def reset_then_init_then_train(seed, the_data, hyperparams, configData, sets_in_use, target_columns_in_use):
	import tensorflow as tf

	DEBUG = ''
	assert hyperparams['run_baseline']

	####################################################
	# === Prepare the data
	_x_cal = the_data['_x_cal']
	_y_cal = the_data['_y_cal']
	_x_val = the_data.get('_x_val')
	_y_val = the_data.get('_y_val')
	_x_test = the_data['_x_test']
	_y_test = the_data['_y_test']

	# === Prepare the hyperparams
	n_epochs = hyperparams['n_epochs']
	conv_filter_init = hyperparams['conv_filter_init']
	FC_init = hyperparams['FC_init']
	do_ES = hyperparams['do_ES']

	# === Seed
	tf.keras.backend.clear_session()
	if (seed is not None):
		set_all_seeds(seed)

	########### DEFINE HYPERPARAMETERS AND INSTANTIATE THE MODEL ###################
	BATCH=128 

	# default to baseline hyperparams:
	if (conv_filter_init is None):
		conv_filter_init = 'he_normal'
	else:
		warn = f'WARNING: conv_filter_init of baseline model overridden with {conv_filter_init}'; DEBUG += warn+'\n';

	if (FC_init is None):
		FC_init = 'he_normal'
	else:
		warn = f'WARNING: FC_init of baseline model overridden with {FC_init}'; DEBUG += warn+'\n';

	if (n_epochs is None):
		n_epochs = 750
	else:
		warn = f'WARNING: n_epochs of baseline model overridden with {n_epochs}'; DEBUG += warn+'\n';

	assert hyperparams['LR'] is None
	assert hyperparams['conv_filter_width'] is None
	assert hyperparams['conv_L2_reg_scale'] is None
	assert hyperparams['FC_L2_reg_scale'] is None
	assert hyperparams['batch_size'] is None

	hyperparams = {
		'conv_filter_width': 21,
		'conv_L2_reg_scale': 0.011/2,
		'FC_L2_reg_scale': 0.011/2,

		'conv_filter_init': conv_filter_init,
		'FC_init': FC_init,
	}

	model_cnn=init_model(hyperparams, _x_cal, seed)

	########### COMPILE MODEL WITH ADAM OPTIMIZER #####################################
	LR=0.01*BATCH/256.

	optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

	## Compile the model defining the optimizer, the loss function and the metrics to track during training
	model_cnn.compile(optimizer=optimizer, loss='mse', metrics=['mse'])  

	########### DEFINE USEFUL CALLBACK FUNCTIONS #####################################

	callbacks = []

	print('Reduce learning rate dynamically')
	rdlr = ReduceLROnPlateau(patience=25, factor=0.5, min_lr=1e-6, monitor='val_loss', verbose=0)
	callbacks.append(rdlr)

	if (do_ES):
		print('EarlyStopping enabled')
		early_stop = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=50, mode='auto', restore_best_weights=True)
		callbacks.append(early_stop)
	else:
		msg='EarlyStopping DISABLED'; print(msg); DEBUG += msg+'\n';

	########### TRAIN THE MODEL #####################################################
	## Train the model
	h=model_cnn.fit(_x_cal, 
	                _y_cal, 
	                batch_size=BATCH, 
	                epochs=n_epochs,
	                validation_data=None if (_x_val is None) else (_x_val, _y_val),
	                callbacks=callbacks,
	                verbose=0)

	
	### RETURN RESULTS
	return_dict = {}

	xy = {
		'train': {'x': _x_cal, 'y': _y_cal},
		'test':  {'x': _x_test, 'y': _y_test},
	}
	if ('dev' in sets_in_use):
		xy['dev'] = {'x': _x_val, 'y': _y_val}

	# save predictions (follows same structure as `ensembling.save_test_set_predictions`)
	dfs = []
	for which_set in sets_in_use:
		predictions = model_cnn.predict(xy[which_set]['x'])
		rmse  = np.sqrt(mean_squared_error(xy[which_set]['y'], predictions))

		return_dict[f'RMSE_{which_set}'] = rmse
		xy[which_set]['y_pred'] = predictions

		df = pd.DataFrame({'DM':xy[which_set]['y'][:,0], 'DM_pred':predictions[:,0]})
		df['set'] = which_set
		dfs.append(df)

	# this df is like `Ensembler.metadata_of_models[nth_training_run]['predictions_df']`
	predictions_df = pd.concat(dfs, axis=0)

	return_dict['seed'] = seed
	return_dict['h.history'] = h.history
	return_dict['predictions_df'] = predictions_df
	return_dict['DEBUG'] = DEBUG
	

	return return_dict