#!/usr/bin/env python3
""" Program to run hyperparam optimization via hyperopt.
Date: 2021-01-20
"""

from hyperopt import fmin, tpe, rand, Trials
from hyperopt.mongoexp import MongoTrials
import hyperopt
import time
import pickle
import pandas as pd
import numpy as np
import subprocess
from os.path import join, exists
import os
import datetime
import logging
from pprint import pprint
from hyperopt.pyll.stochastic import sample
import matplotlib.pyplot as plt
import atexit
from functools import partial

from components.hyperopt_config.objective import calc_loss
from components.hyperopt_config.get_cmd_space import get_cmd_space
from components.console_logging import setup_logging, end_logging

__author__ = "Matthew Dirks"

DEFAULT_DB_PORT = 27017
MONGO_EXPERIMENT_NAME = 'experiment'


def main_master(experiment_name, max_evals, DB_host='127.0.0.1', DB_port=None, out_dir='.', which_cmd_space=None):
	global server

	assert which_cmd_space is not None

	print(f'run_hyperopt.py: running main_master({experiment_name}, {max_evals}, {DB_host}, {DB_port})')
	logger = setup_logging(out_dir, suffix='_hyperopt_master', logger_to_use=logging.getLogger('hyperopt_master'))
	logger.info('==================== RUNNING HYPEROPT MASTER ====================')

	DB_NAME = experiment_name

	if (DB_port is None):
		DB_port = DEFAULT_DB_PORT

	logger.info(f'which_cmd_space = {which_cmd_space}')
	cmd_space, hyperhyperparams = get_cmd_space(which_cmd_space)

	connect_url = f'mongo://{DB_host}:{DB_port}/{DB_NAME}/jobs'
	logger.info(f'Connecting to DB at {connect_url}, using experiment name {MONGO_EXPERIMENT_NAME}')

	trials = MongoTrials(connect_url, exp_key=MONGO_EXPERIMENT_NAME)

	logger.info('NOTE: next you must run a worker, because next line calls fmin which will block until a worker does all the jobs.')
	logger.info('MASTER starting now...{}'.format(datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S")))

	f = lambda sample_of_cmd_space: calc_loss(sample_of_cmd_space, hyperhyperparams)

	algo = partial(tpe.suggest, n_EI_candidates=50) # default n_EI_candidates is 24
	# ^ see https://github.com/hyperopt/hyperopt/issues/632
	best = fmin(f, space=cmd_space, algo=tpe.suggest, max_evals=max_evals, trials=trials, verbose=True)

	try:
		logger.info(f'best: {best}')
	except Exception as e:
		logger.error(f'[1] {e}')

	try:
		logger.info('losses: ', trials.losses())
	except Exception as e:
		logger.error(f'[2] {e}')

	try:
		logger.info('len(results): ', len(trials.results))
	except Exception as e:
		logger.error(f'[3] {e}')


if __name__ == '__main__':
	@atexit.register
	def goodbye():
		global server

		logger = logging.getLogger('hyperopt_master')
		logger.info('Exit-handler running - goodbye!')

		end_logging(logger)

		print('DONE')

	import fire
	fire.Fire(main_master)
