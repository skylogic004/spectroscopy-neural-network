from os.path import join, split, realpath, dirname, splitext, isfile, exists, isdir
import json
from typing import Callable

import components.colorama_helpers as _c

def strip_quotes(s):
	if (s.startswith('"') and s.endswith('"')) or (s.startswith('\'') and s.endswith('\'')):
		return s[1:-1]
	else:
		return s

def load_cmd_args_file(cmd_args_fpath):
	cmd_args_fpath = strip_quotes(cmd_args_fpath)
	assert exists(cmd_args_fpath), 'Provided cmd_args_fpath ({}) doesn\'t exist'.format(cmd_args_fpath)

	err = ''
	ext = splitext(cmd_args_fpath)[1]
	if (ext == '.json'):
		with open(cmd_args_fpath, 'r') as f:
			cmd_args = json.load(f)
	elif (ext == '.pyon'):
		from components.config_io import config_pyon
		cmd_args, err = config_pyon.read(cmd_args_fpath)

	if (cmd_args is None):
		raise(Exception('Failed to load cmd args from file. cmd_args_fpath="{}", err="{}"'.format(cmd_args_fpath, err)))

	return cmd_args

# Callable[..., str] is roughly equivalent to Callable[[VarArg(), KwArg()], int]

def get_main_function(run_function:Callable[..., str]):
	def main(*positional_args, **override_args):
		#### check if cmd_args_fpath provided
		cmd_args_fpath = None

		n_pos = len(positional_args)
		if (n_pos == 1 and 'cmd_args_fpath' not in override_args):
			# assume first positional arg is cmd_args_fpath
			cmd_args_fpath = positional_args[0]
		elif (n_pos == 0):
			if ('cmd_args_fpath' in override_args):
				cmd_args_fpath = override_args['cmd_args_fpath']
		else:
			print(_c.red('Invalid command line arguments: please specify either (a) cmd_args_fpath (positional or kwarg) and with or without additional kwargs to override those specified in cmd_args_fpath, or (b) only kwarg args (make sure all args start with "--")'))
			print(_c.red('len(positional_args) = {}\nlen(override_args) = {} (kwargs)'.format(n_pos, len(override_args))))
			exit()

		if (cmd_args_fpath is not None): 
			# if cmd_args_fpath provided, then load the config file with command arguments
			cmd_args = load_cmd_args_file(cmd_args_fpath)

			# modify the args given in the file with arguments manually specified (overrides settings from config file)
			cmd_args.update(override_args)

			if ('cmd_args_fpath' not in cmd_args):
				# if a positional arg is used to specify cmd_args_fpath rather than a kwarg,
				# then add it into the cmd_args dictionary here
				cmd_args['cmd_args_fpath'] = cmd_args_fpath
		else: 
			# no config file, so read command line args only
			cmd_args = override_args

		#### pass on args to build_and_train function
		return run_function(**cmd_args)

	return main