import logging
import os
import traceback
from os.path import join
import sys

import components.colorama_helpers as _c

def setup_logging(outDir, suffix='', logger_to_use=None):
	if (logger_to_use is None):
		logger = logging.getLogger()
	else:
		logger = logger_to_use

	# remove existing handlers, if any
	for handler in logger.handlers[:]:  # make a copy of the list
		logger.removeHandler(handler)
			
	logger.setLevel(logging.DEBUG)

	# logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s")
	logFormatter = logging.Formatter("%(message)s")

	fileHandler = logging.FileHandler(join(outDir, 'consoleOutput%s.txt' % suffix))
	fileHandler.setFormatter(logFormatter)
	fileHandler.setLevel(logging.DEBUG)
	logger.addHandler(fileHandler)

	consoleHandler = logging.StreamHandler()
	consoleHandler.setFormatter(logFormatter)
	consoleHandler.setLevel(logging.DEBUG)
	logger.addHandler(consoleHandler)

	### Also log uncaught exceptions to disk (and console)
	def uncaught_exception_handler(type, exception_value, tb):
		logger.exception(_c.red('\nUncaught exception: {}, {}\n'.format(type, exception_value)))
		logger.debug('Exception traceback details:')
		logger.debug(''.join(traceback.format_exception(etype=type(exception_value), value=exception_value, tb=exception_value.__traceback__)))


	# Install exception handler
	sys.excepthook = uncaught_exception_handler

	return logger

def end_logging(logger_to_use=None):
	if (logger_to_use is None):
		logger = logging.getLogger()
	else:
		logger = logger_to_use

	handlers = logger.handlers[:]
	for handler in handlers:
		handler.close()
		logger.removeHandler(handler)