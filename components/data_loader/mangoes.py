import logging
import pandas as pd
from sklearn.utils import shuffle
from colorama import Fore, Back, Style
from collections import OrderedDict
from scipy.signal import savgol_filter
import numpy as np
import datetime
import random
import math

CSV_FPATH = "data/NAnderson2020MendeleyMangoNIRData.csv"
MAT_FPATH = "data/mango_dm_full_outlier_removed2.mat"
METADATA_FPATH = "data/metadata.pkl"
RANDOM_STATE = 5000

def load_Anderson_data():
	logger = logging.getLogger()
	logger.info(Fore.CYAN + 'Dataset: Mango' + Fore.RESET)

	# dictionary for saving stuff later
	data_dict = OrderedDict()

	# load each sensor
	df, feature_columns, targets_to_use = load_Anderson_data_helper()

	assay_columns_dict = {x:x for x in targets_to_use}

	# random shuffle
	shuffled_df = shuffle(df, random_state=RANDOM_STATE).copy()
	shuffled_df.reset_index(drop=False, inplace=True) # let index be row count from 0 onward (in the new shuffled state)

	# not in use
	shuffled_df['ignore'] = False

	data_dict['shuffled_df'] = shuffled_df

	return {
		'data_dict': data_dict, 
		'assay_columns_dict': assay_columns_dict, 
		'feature_columns': feature_columns,
		'extra_columns': ['Date'],
	}

def load_Anderson_data_helper():
	# spectra
	df = pd.read_csv(CSV_FPATH)
	nir_columns = df.columns[list(df.columns).index('285'):]
	assert len(nir_columns) == 306

	# assays
	targets_to_use = ['DM']

	# original dataset split into sets -- rename the values
	mapper = {'Cal': 'calibrate', 'Tuning': 'tuning', 'Val Ext': 'test'}
	df['rand_split'] = df['Set'].apply(mapper.get)

	# create new validation split(s)
	df = make_new_validation_split(df, is_Dario_dataset=False)

	# Paper says they use 103 features but there are 306 bins in the spectra. They said they use 684 to 990 nm, so lets truncate to match
	wavelengths = [int(x) for x in nir_columns]
	wavelengths_truncated = [x for x in wavelengths if (x >= 684 and x <= 990)]
	nir_columns_truncated = [str(x) for x in wavelengths_truncated]

	# Meta-data: what are the column names
	feature_columns = OrderedDict()
	feature_columns['NIR'] = nir_columns
	feature_columns['NIR_truncated'] = nir_columns_truncated

	# Pre-processing
	df, feature_columns = do_preprocessing(feature_columns, nir_columns_truncated, df)

	return df, feature_columns, targets_to_use

def snv(spectrum):
	"""
	See https://towardsdatascience.com/scatter-correction-and-outlier-detection-in-nir-spectroscopy-7ec924af668,

	:snv: A correction technique which is done on each
	individual spectrum, a reference spectrum is not
	required
	
	return:
		Scatter corrected spectra
	"""
	return (spectrum - np.mean(spectrum)) / np.std(spectrum)

def do_preprocessing(feature_columns, nir_columns_truncated, df):
	# SNV
	snv_df = df[nir_columns_truncated].apply(snv, axis=1).rename(columns={col:f'SNV_{col}' for col in nir_columns_truncated})
	feature_columns['SNV'] = snv_df.columns.tolist()

	# 1st DERIVATIVE
	window_length = 13
	polyorder = 2

	deriv = 1
	columns = [f'SG{deriv}_{col}' for col in nir_columns_truncated]
	feature_columns[f'SG{deriv}'] = columns

	sg_1stderiv_df = pd.DataFrame(df[nir_columns_truncated].apply(lambda row: savgol_filter(row, window_length, polyorder, deriv=deriv), axis=1).tolist(), columns=columns)

	# 2nd DERIVATIVE
	deriv = 2
	columns = [f'SG{deriv}_{col}' for col in nir_columns_truncated]
	feature_columns[f'SG{deriv}'] = columns

	sg_2ndderiv_df = pd.DataFrame(df[nir_columns_truncated].apply(lambda row: savgol_filter(row, window_length, polyorder, deriv=deriv), axis=1).tolist(), columns=columns)


	# 1st DERIVATIVE on SNV
	deriv = 1
	columns = [f'SNV_SG{deriv}_{col}' for col in nir_columns_truncated]
	feature_columns[f'SNV_SG{deriv}'] = columns

	snv_sg_1stderiv_df = pd.DataFrame(snv_df.apply(lambda row: savgol_filter(row, window_length, polyorder, deriv=deriv), axis=1).tolist(), columns=columns)

	# 2nd DERIVATIVE on SNV
	deriv = 2
	columns = [f'SNV_SG{deriv}_{col}' for col in nir_columns_truncated]
	feature_columns[f'SNV_SG{deriv}'] = columns

	snv_sg_2ndderiv_df = pd.DataFrame(snv_df.apply(lambda row: savgol_filter(row, window_length, polyorder, deriv=deriv), axis=1).tolist(), columns=columns)

	# combine dfs
	combined_df = pd.concat([df, snv_df, sg_1stderiv_df, sg_2ndderiv_df, snv_sg_1stderiv_df, snv_sg_2ndderiv_df], axis=1)
	assert df.shape[0] == combined_df.shape[0] == snv_df.shape[0] == snv_sg_2ndderiv_df.shape[0]

	# check sizes
	total = 0
	for key, value in feature_columns.items():
		print(key, len(value))
		total += len(value)
	print('Total number of features: ', total)

	return combined_df, feature_columns

def load_Dario_data():
	""" Most of this based on https://github.com/dario-passos/DeepLearning_for_VIS-NIR_Spectra/blob/master/notebooks/Tutorial_on_DL_optimization/1)%20optimization_tutorial_regression.ipynb """

	from scipy.io import loadmat
	from sklearn.model_selection import train_test_split
	from sklearn.preprocessing import StandardScaler 

	logger = logging.getLogger()
	logger.info(Fore.CYAN + 'Dataset: Mangoes (Dario\'s)' + Fore.RESET)

	# dictionary for saving stuff later
	data_dict = OrderedDict()

	# load Dario's matlab data
	data = loadmat(MAT_FPATH)
	logger.info(f'Keys in .mat file: {data.keys()}')

	# these are the available keys:
	# 'wave'         : wavelengths
	# 'DM_cal'       : target, *train* set (mislabelled as "cal")
	# 'DM_test'      : target, test set
	# 'Sp_cal'       : spectra, *train* set (mislabelled as "cal")
	# 'Sp_test'      : spectra, test set
	# 'SP_all_train' : spectra + preProcessedSpectra, train set
	# 'SP_all_test'  : spectra + preProcessedSpectra, test set
	#
	# They split the whole dataset into 2 parts: train and test.
	# Then they split `train` into 2 parts: `cal` (calibration) and `tuning` (aka "dev" or "validation") 

	## Convert data type to float32 for better inter-operability with TensorFlow
	x_train=data['SP_all_train'].astype(np.float32)
	y_train=data['DM_cal'].astype(np.float32)
	x_test=data['SP_all_test'].astype(np.float32)
	y_test=data['DM_test'].astype(np.float32)

	## Spliting the full train set into calib and tuning subsets. It is important to set the 'random_state'
	## parameter to a fixed value in order to guarentee that each time you run the experiment, the data is
	## split the same way
	x_cal, x_tuning, y_cal, y_tuning = train_test_split(x_train, y_train, test_size=0.33, shuffle=True, random_state=42) 

	## The wavelenghts for the XX axis when we plot the spectra
	x_scale=data['wave'].astype(np.float32).reshape(-1,1)

	## Check for dimensions
	logger.info('Data set dimensions ----------------------------')
	logger.info('Full Train set dims X Y = {}\t{}'.format(x_train.shape, y_train.shape))
	logger.info('Calibration set dims X Y = {}\t{}'.format(x_cal.shape, y_cal.shape))
	logger.info('Tuning set dims X Y = {}\t{}'.format(x_tuning.shape, y_tuning.shape))
	logger.info('Test set dims X Y = {}\t{}'.format(x_test.shape, y_test.shape))
	logger.info('wavelengths number = {}'.format(np.shape(x_scale)))

	## Since the test set is unknown (we are not suppose to have access to it during the
	## optimization of the model) the scalling process should take this into account. We
	## have to define a scaler based only on the train data, and apply it to the test data.

	def standardize_column(_x_train, _x_cal, _x_tuning, _x_test):
		## We train the scaler on the full train set and apply it to the other datasets
		scaler = StandardScaler().fit(_x_train)
		## for columns we fit the scaler to the train set and apply it to the test set
		x_cal_scaled = scaler.transform(_x_cal)
		x_tuning_scaled = scaler.transform(_x_tuning)
		x_test_scaled = scaler.transform(_x_test)
		return x_cal_scaled, x_tuning_scaled, x_test_scaled

	## Standardize on columns on the spectra in train set
	# (and recall that train = cal + tuning)
	x_cal_scaled, x_tuning_scaled, x_test_scaled = standardize_column(x_train, x_cal, x_tuning, x_test)

	# concat back into one big array, then put into DataFrame
	cal    = np.concatenate((x_cal_scaled, y_cal), axis=1)
	tuning = np.concatenate((x_tuning_scaled, y_tuning), axis=1)
	test   = np.concatenate((x_test_scaled, y_test), axis=1)

	# label the original spectrum columns with their wavelength:
	wavelength_columns = [f'wave{w:03d}' for w in data['wave'][0]]

	# label the pre-processed spectra with arbitrary numbers
	spectrum_columns = wavelength_columns + [f'sp{x:03d}' for x in range(x_cal_scaled.shape[1] - len(wavelength_columns))]

	# append target column
	target = 'DM'
	column_names = spectrum_columns + [target]

	# convert to DF
	cal_df    = pd.DataFrame(cal, columns=column_names)
	tuning_df = pd.DataFrame(tuning, columns=column_names)
	test_df   = pd.DataFrame(test, columns=column_names)

	cal_df['rand_split'] = 'calibrate'
	tuning_df['rand_split'] = 'dev'
	test_df['rand_split'] = 'test'

	outliers_removed_df = pd.concat([cal_df, tuning_df, test_df], axis=0).reset_index(drop=True)
	outliers_removed_df['sample_id'] = np.arange(outliers_removed_df.shape[0])

	# join with metadata df
	# this has the Set, Season, Region, Date, Type, Cultivar, etc columns taken from `NAnderson2020MendeleyMangoNIRData.csv`
	metadata_df = pd.read_pickle(METADATA_FPATH)
	# don't need these (they were just in metadata_df to help confirm a proper a join)
	metadata_df.drop(columns=['687', '789'], inplace=True)

	joined_df = outliers_removed_df.merge(metadata_df, on=['sample_id','DM'])
	assert joined_df.shape[0] == metadata_df.shape[0] == outliers_removed_df.shape[0]

	### then do new val splits here.
	joined_df = make_new_validation_split(joined_df, is_Dario_dataset=True)

	# pass on the column names of the "input features" (possible inputs to neural network)
	feature_columns = OrderedDict()
	feature_columns['NIR_preprocessed_and_outliers_removed'] = spectrum_columns

	# pass on the column names of the "targets" (possible outputs of neural network)
	assay_columns_dict = {target:target}

	# specify which samples to ignore (not in use)
	joined_df['ignore'] = False

	data_dict['shuffled_df'] = joined_df
	return {
		'data_dict': data_dict, 
		'assay_columns_dict': assay_columns_dict, 
		'feature_columns': feature_columns,
		'extra_columns': ['Date','sample_id'],
	}

def make_new_validation_split(df, is_Dario_dataset):
	orig_counts = df['rand_split'].value_counts()
	N_VAL = orig_counts['dev']

	df['datetime'] = pd.to_datetime(df['Date'])

	############## 2017v3 ###################################################################
	season_to_set = {1:'calibrate', 2:'calibrate', 3:'validate', 4:'test'}
	df.loc[:, 'D2017_split'] = df['Season'].apply(season_to_set.get)

	############## 2017v4 ###################################################################
	### reduce points to the same size as original validation set (and grow the training set)
	if (is_Dario_dataset):
		# I'll take everything between 9-28 and start of test set...
		sep28_up_to_test_set_mask = (df['datetime'] >= pd.Timestamp(datetime.date(2017,9,28)))
		sep28_up_to_test_set_mask &= ~(df['rand_split']=='test')
		# this gives 2845 spectra (for the validation set, target is 3272)
		# but that's not enough, so I'll take SOME of the samples from 9-27, until I have enough
		need_n_more_spectra = N_VAL - sep28_up_to_test_set_mask.sum()
		# there seem to be 2 spectra per unique DM value (one DM per mango, I presume)
		need_n_more_mangoes = need_n_more_spectra / 2

		sep27_mask = (df['datetime'] >= pd.Timestamp(datetime.date(2017,9,27)))
		sep27_mask &= (df['datetime'] < pd.Timestamp(datetime.date(2017,9,28)))

		random.seed(44002)
		random_mangoes_DMs = random.sample(df[sep27_mask]['DM'].unique().tolist(), math.ceil(need_n_more_mangoes))

		sample_from_sep27_mask = (df['DM'].isin(random_mangoes_DMs)) & sep27_mask
		#print(len(random_mangoes_DMs), 'unique DMs selected, leading to', sample_from_sep27_mask.sum(), 'spectra')
		#> 214 unique DMs selected, leading to 427 spectra

		new_val_mask = sample_from_sep27_mask | sep28_up_to_test_set_mask
		# new size = 3272 (matches my target exactly)

		new_name = 'D3_split'
		df.loc[:, new_name] = 'calibrate'
		df.loc[new_val_mask, new_name] = 'tuning'
		df.loc[df['rand_split']=='test', new_name] = 'test'

	############## 2015 ###################################################################
	### same size as original train and validation sets,
	### but validation set is all samples from 2015
	### (and 420 spectra from 2015 harvest season are left in the calibration set)
	if (is_Dario_dataset):
		# take first 3272 spectra...
		datetime_of_last = df.sort_values('datetime').iloc[:N_VAL].iloc[-1]['datetime']
		# this date is Timestamp('2016-01-02 00:00:00')

		# Taking up to and including the date above gave 3327 spectra (too many). 
		# So instead I'll start with not enough spectra and add some after...
		the_earliest_samples_mask = df['datetime'] < datetime_of_last

		# this is 3173 spectra. but that's not enough, so I'll take some more
		need_n_more_spectra = N_VAL - the_earliest_samples_mask.sum()

		# there's usually 2 spectra per unique DM value (one DM per mango, I presume)
		need_n_more_mangoes = need_n_more_spectra / 2
		
		# The next date after the ones I already selected is Jan 2. I'll pull some spectra from here.
		jan2_mask = (df['datetime'] >= datetime_of_last)
		jan2_mask &= (df['datetime'] < pd.Timestamp(datetime.date(2016,1,3)))

		# turns out there's no way to get exactly 99 spectra... 
		# because there happens to be 2 per mango, for every mango, this time.
		# I guess I'll just split one mango up
		random.seed(44000)
		random_mangoes_DMs = random.sample(df[jan2_mask]['DM'].unique().tolist(), math.ceil(need_n_more_mangoes))

		sample_from_jan2_mask = (df['DM'].isin(random_mangoes_DMs)) & jan2_mask
		#print(len(random_mangoes_DMs), 'unique DMs selected, leading to', sample_from_jan2_mask.sum(), 'spectra')
		# 50 unique DMs selected, leading to 100 spectra

		one_random_spectrum_id = random.sample(df[sample_from_jan2_mask]['sample_id'].tolist(), 1)

		sample_from_jan2_mask2 = (df['DM'].isin(random_mangoes_DMs)) & jan2_mask & (df['sample_id'] != one_random_spectrum_id[0])
		#print(len(random_mangoes_DMs), 'unique DMs selected, leading to', sample_from_jan2_mask2.sum(), 'spectra')
		# 50 unique DMs selected, leading to 99 spectra

		# ok, keep these selected...
		new_val_mask = the_earliest_samples_mask | sample_from_jan2_mask2

		# Save the new split to df
		new_name = 'D1_split'
		df.loc[:, new_name] = 'calibrate'
		df.loc[new_val_mask, new_name] = 'tuning'
		df.loc[df['rand_split']=='test', new_name] = 'test'


	############## 2016 ###################################################################
	### same size as original train and validation sets,
	### but validation set samples from 2016 *plus* a bunch of samples from the end of 2015 and from  the beginning of 2017
	### (because 2016 has much fewer spectra samples than the others)
	### This is the spectra counts from each season: {1: 3692, 2: 1353, 3: 4869, 4: 1448}
	### Also note: original split's validation is 33% of non-test samples (not 33.3repeating) which means some are some samples 
	### (I guess about 0.66666% of samples) that NEVER participate in validation set,
	### i.e. they are not in the validation sets in ANY of these: D1_split, D2_split, D3_split
	if (is_Dario_dataset):
		# If I took the remaining samples, after removing D3_split dev set and D1_split dev set...
		mask1 = df['D3_split']=='tuning'
		mask2 = df['D1_split']=='tuning'

		# There shouldn't be overlap:
		assert (mask1 & mask2).sum() == 0

		# these are samples that are left when you remove 2015 dev set and 2017 dev set (test set is still here, I remove it later)
		new_train_set_mask = mask1 | mask2

		# select the first date in samples deemed dev set, and the last date
		selected_day1 = (df['datetime'] >= pd.Timestamp(datetime.date(2016,1,2)))
		selected_day1 &= (df['datetime'] < pd.Timestamp(datetime.date(2016,1,3)))
		selected_day2 = (df['datetime'] >= pd.Timestamp(datetime.date(2017,9,27)))
		selected_day2 &= (df['datetime'] < pd.Timestamp(datetime.date(2017,9,28)))
		choose_from_these_mask = ((selected_day1 | selected_day2) & (new_train_set_mask==False)) # only the dev set samples from the 2 days

		# from these 2 days, randomly select some to add to training set
		n_samples_needed = orig_counts['calibrate'] - new_train_set_mask.sum()
		need_n_more_mangoes = n_samples_needed / 2

		# I found that this random seed gets the right number of spectra (because not all mangoes have exactly 2 spectra)
		random.seed(6)

		# select n mangoes at random (where a "mango" is actually just a set of spectra with unique DM)
		# (n is `need_n_more_mangoes`)
		random_mangoes_DMs = random.sample(df[choose_from_these_mask]['DM'].unique().tolist(), math.ceil(need_n_more_mangoes))
		chosen_mask = (df['DM'].isin(random_mangoes_DMs)) & choose_from_these_mask

		# final mask (for training set) is the randomly selected samples from the first and last days within dev set combined with the new training set from before
		final_train_mask = chosen_mask | new_train_set_mask

		# Save the new split to df
		new_name = 'D2_split'
		df.loc[:, new_name] = 'tuning'
		df.loc[final_train_mask, new_name] = 'calibrate'
		df.loc[df['rand_split']=='test', new_name] = 'test'

	############## train5045_random_split ###################################################################
	if (is_Dario_dataset):
		# here, select non-test samples. The random split will be from amongst these.
		choose_from_these_mask = df['rand_split'] != 'test'
		assert df.shape[0] - 1448 == choose_from_these_mask.sum()

		# this is how many spectra we want to be in the new training set:
		sizes = df['D2017_split'].value_counts()
		n_samples_needed = sizes['calibrate']

		# I assume that unique DM values correspond to unique mangoes
		# (I'm certain that 1 mango cannot have 2 DM values)
		# (but there may be 2 mangoes with equal DM values, in which case, those 2 mangoes will "travel" together, and that's fine)
		unique_DM_values = df[choose_from_these_mask]['DM'].unique()

		# found that seed=0 and n=2005 works to find the right number of spectra
		random.seed(136)
		n = 2012

		# select n mangoes at random (where a "mango" is actually just a set of spectra with unique DM)
		random_mangoes_DMs = random.sample(unique_DM_values.tolist(), n)

		chosen_mask = (df['DM'].isin(random_mangoes_DMs)) & choose_from_these_mask

		assert (chosen_mask.sum() == n_samples_needed)

		new_name = 'train5045_random_split'
		df.loc[:, new_name] = 'tuning'
		df.loc[chosen_mask, new_name] = 'calibrate'
		df.loc[df['rand_split']=='test', new_name] = 'test'

	return df