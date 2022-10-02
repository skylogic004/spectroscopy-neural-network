"""
This python module allows creating a custom NN architecture (dynamically).
Code follows a similar structure to `baseline_model.py`.
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


def init_model(hyperparams, input_size, seed=None):
	logger = logging.getLogger()

	conv_filter_init = hyperparams['conv_filter_init']
	conv_filter_width = hyperparams['conv_filter_width']
	conv_L2_reg_scale = hyperparams['conv_L2_reg_scale']
	conv_n_filters = hyperparams['conv_n_filters']

	FC_init = hyperparams['FC_init']
	FC_L2_reg_scale = hyperparams['FC_L2_reg_scale']
	FC_layer_sizes = hyperparams['FC_layer_sizes']

	## Layers dimensions
	INPUT_DIMS = input_size
	K_NUMBER = conv_n_filters

	K_WIDTH = conv_filter_width
	K_STRIDE = 1

	K_INIT_FN = get_init_fn(conv_filter_init, seed)
	FC_INIT_FN = get_init_fn(FC_init, seed)

	K_REG = tf.keras.regularizers.l2(conv_L2_reg_scale)
	FC_REG = tf.keras.regularizers.l2(FC_L2_reg_scale)

	model_cnn = keras.Sequential()

	model_cnn.add(keras.layers.Reshape((INPUT_DIMS, 1),input_shape=(INPUT_DIMS,)),)

	model_cnn.add(keras.layers.Conv1D(filters=K_NUMBER, 
							kernel_size=K_WIDTH, 
							strides=K_STRIDE, 
							padding='same', 
							kernel_initializer=K_INIT_FN,
							kernel_regularizer=K_REG,
							activation='elu',
							input_shape=(INPUT_DIMS,1)))
	model_cnn.add(keras.layers.Flatten())

	for size in FC_layer_sizes:
		model_cnn.add(keras.layers.Dense(size, 
										 kernel_initializer=FC_INIT_FN, 
										 kernel_regularizer=FC_REG, 
										 activation='elu'))

	model_cnn.add(keras.layers.Dense(1, 
									 kernel_initializer=FC_INIT_FN, 
									 kernel_regularizer=FC_REG, 
									 activation='linear'))

	return(model_cnn)


def reset_then_init_then_train(seed, the_data, hyperparams, configData, sets_in_use, target_columns_in_use):
	import tensorflow as tf

	DEBUG = ''

	####################################################
	# === Prepare the data
	_x_cal = the_data['_x_cal']
	_y_cal = the_data['_y_cal']
	_x_val = the_data.get('_x_val')
	_y_val = the_data.get('_y_val')
	_x_test = the_data['_x_test']
	_y_test = the_data['_y_test']

	# === Prepare hyperparams
	LR_sched_settings = hyperparams['LR_sched_settings']
	do_ES = hyperparams['do_ES']

	# === Seed
	tf.keras.backend.clear_session()
	if (seed is not None):
		set_all_seeds(seed)

	########### DEFINE HYPERPARAMETERS AND INSTANTIATE THE MODEL ###################
	input_size = np.shape(_x_cal)[1]
	model_cnn=init_model(hyperparams, input_size, seed)

	n_epochs = hyperparams['n_epochs']

	########### COMPILE MODEL WITH ADAM OPTIMIZER #####################################
	## Compile the model defining the optimizer, the loss function and the metrics to track during training
	model_cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparams['LR']), loss='mse', metrics=['mse'])  

	callbacks = []

	assert isinstance(LR_sched_settings, dict) or LR_sched_settings == 'off'

	if (LR_sched_settings == 'off'):
		msg='LR schedule: off'; print(msg); DEBUG += msg+'\n';
	elif (LR_sched_settings['type'] == 'ReduceLROnPlateau'):
		scheduler_args = {
			'patience': LR_sched_settings.get('patience', 25),
			'factor': LR_sched_settings.get('factor', 0.5),
			'min_lr': LR_sched_settings.get('min_lr', 1e-6),
		}
		msg=f'LR schedule: ReduceLROnPlateau. Parameters = {scheduler_args}'; print(msg); DEBUG += msg+'\n';
		rdlr = ReduceLROnPlateau(**scheduler_args, monitor='val_loss', verbose=0)
		callbacks.append(rdlr)
	elif (LR_sched_settings['type'] == 'drop_LR_at_epochs'):
		msg='LR schedule: Custom callback, drops at epochs specified.'; print(msg); DEBUG += msg+'\n';

		drop_LR_at_epochs = LR_sched_settings['drop_LR_at_epochs']

		assert isinstance(drop_LR_at_epochs, list)
		assert len(drop_LR_at_epochs) > 0
		

		def scheduler(epoch, lr):
			""" drop LR in half at the epochs specified in `drop_LR_at_epochs` list """
			#print(epoch, lr, epoch in drop_LR_at_epochs)
			if epoch in drop_LR_at_epochs:
				return lr/2
			else:
				return lr
		LR_sched_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
		callbacks.append(LR_sched_callback)

	else:
		raise(ValueError('Invalid LR_sched_settings'))

	if (do_ES):
		msg='EarlyStopping ENABLED'; print(msg); DEBUG += msg+'\n';
		early_stop = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=50, mode='auto', restore_best_weights=True)
		callbacks.append(early_stop)
	else:
		msg='EarlyStopping DISABLED'; print(msg); DEBUG += msg+'\n';

	########### TRAIN THE MODEL #####################################################
	## Train the model

	h = model_cnn.fit(_x_cal, 
					  _y_cal, 
					  batch_size=hyperparams['batch_size'], 
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

		try:
			rmse  = np.sqrt(mean_squared_error(xy[which_set]['y'], predictions))
		except:
			rmse = 'Exception - no value'

		return_dict[f'RMSE_{which_set}'] = rmse
		xy[which_set]['y_pred'] = predictions

		assert len(target_columns_in_use)==1, 'This code written for one target only'
		target = target_columns_in_use[0]

		df = pd.DataFrame({target:xy[which_set]['y'][:,0], f'{target}_pred':predictions[:,0]})
		df['set'] = which_set
		dfs.append(df)

	# this df is like `Ensembler.metadata_of_models[nth_training_run]['predictions_df']`
	predictions_df = pd.concat(dfs, axis=0)

	return_dict['seed'] = seed
	return_dict['h.history'] = h.history
	return_dict['predictions_df'] = predictions_df
	return_dict['DEBUG'] = DEBUG
	
	return return_dict