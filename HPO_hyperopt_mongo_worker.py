import os
import sys
from hyperopt import mongoexp
import datetime
import logging

from HPO_start_master import DEFAULT_DB_PORT

__author__ = "Matthew Dirks"

def main(experiment_name, DB_host='127.0.0.1', DB_port=None, n_jobs=sys.maxsize, timeout_hours=None):
	print(f'HPO_hyperopt_mongo_worker.py main({experiment_name}, {DB_port}, {DB_host})...')
	assert os.environ.get('OVERRIDE_n_gpu') is not None, 'Env var needed, e.g. `set OVERRIDE_n_gpu=1`'

	if (DB_port is None):
		DB_port = DEFAULT_DB_PORT
	print(f'HPO_hyperopt_mongo_worker.py: DB_port is {DB_port}')

	if (timeout_hours is not None):
		timeout_seconds = int(timeout_hours*60*60)
	else:
		timeout_seconds = None


	print('HPO_hyperopt_mongo_worker.py running mongoexp.main()...{}'.format(datetime.datetime.now().strftime("%Y-%m-%d, %H:%M:%S")))
	options = My_Main_Worker_Helper(mongo=f'{DB_host}:{DB_port}/{experiment_name}',
	                                use_subprocesses=False,
	                                last_job_timeout=timeout_seconds,
	                                max_jobs=n_jobs)

	# hyperopt using logging, but I'm circumventing the main entry point so I need to setup the logging module myself:
	logging.basicConfig(stream=sys.stderr, level=logging.INFO)

	mongoexp.main_worker_helper(options, [])


class My_Main_Worker_Helper:
	def __init__(self,
	             exp_key=None,
	             last_job_timeout=None, # no more taking new jobs after this many seconds
	             max_consecutive_failures=4,
	             max_jobs=sys.maxsize,
	             mongo="localhost/hyperopt",
	             poll_interval=5, # seconds. 5 is fine when jobs take many minutes to complete
	             reserve_timeout=120.0,
	             workdir=None,
	             use_subprocesses=True,
	             max_jobs_in_db=sys.maxsize):

		""" This function helps by setting the default values as defined in mongoexp.main_worker,
		also, see that function for documentation on each of these arguments. """

		self.exp_key = exp_key
		self.last_job_timeout = last_job_timeout
		self.max_consecutive_failures = max_consecutive_failures
		self.max_jobs = max_jobs
		self.mongo = mongo
		self.poll_interval = poll_interval
		self.reserve_timeout = reserve_timeout
		self.workdir = workdir
		self.use_subprocesses = use_subprocesses
		self.max_jobs_in_db = max_jobs_in_db

if __name__ == '__main__':
	print('HPO_hyperopt_mongo_worker.py STARTING...')
	import fire
	fire.Fire(main)
