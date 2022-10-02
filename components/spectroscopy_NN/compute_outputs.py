import pandas as pd
import numpy as np

TF_VAR = 'tensorflowVariable'

all_equal = lambda _list: _list.count(_list[0]) == len(_list)

def get_assay_predictions(sess, 
	                      _sets, 
	                      assay_output, 
	                      Xs, 
	                      X_data_dict, 
	                      y_true_assay, 
	                      Y_data_dict,
	                      unnormalize_Y_df,
	                      assay_columns,
	                      cur_data,
	                      extra_columns=[]):

	"""
	Args:
		assay_output: TF Tensor pointing to output of network (yhat)
		Xs: dictionary of TF placeholders (one for each sensor (aka feature) type)
		X_data_dict: dictionary of input data (each is numpy array, normalized)
		y_true_assay: TF placeholder for ground truth
		Y_data_dict: dictionary of ground truth data (each is numpy array, normalized)
		cur_data: *not* normalized
		extra_columns: names of columns from original dataset that we want to keep in the output (example: Channel, pile name, other IDs...)
	"""

	prediction_sets_dfs = {}
	for _set in _sets:
		tmp_feed_dict = {tfvar:X_data_dict[(_set, feature)] for feature, tfvar in Xs.items()}
		tmp_feed_dict[y_true_assay] = Y_data_dict[_set]

		pred = sess.run(assay_output, feed_dict=tmp_feed_dict)

		pred = pd.DataFrame(pred, columns=assay_columns) # keep same column names as input data for now, because unnormalizing requires the column names
		pred = unnormalize_Y_df(pred)
		pred.columns = [x+'_pred' for x in assay_columns] # rename the column, to make it clear these are predictions not ground truth
		pred['set'] = _set # record whether in training, dev, or test sets
		prediction_sets_dfs[_set] = pred

	all_pred = pd.concat([prediction_sets_dfs[_set] for _set in _sets], axis=0)
	all_assays = pd.concat([cur_data.get(_set, filtered=True, combine_train_dev=False).loc[:, assay_columns+extra_columns] for _set in _sets], axis=0)

	# copy the index from the original data, so we can cross-reference samples later
	all_pred.index = all_assays.index
	joined_df = pd.concat([all_assays, all_pred], axis=1)

	return joined_df
