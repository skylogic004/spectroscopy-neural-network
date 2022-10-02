# Neural Network for Spectroscopy

This repo contains the source code used to run the experiments described in **"Automatic Neural Network Hyperparameter Optimization for Extrapolation: Lessons Learned from Visible and Near-Infrared Spectroscopy of Mango Fruit"** by Matthew Dirks and David Poole.

# Software Requirements

The main requirements are:

- Python 3.9.6
- Tensorflow 2.6.0

A complete listing of the python package environment (which uses `pip`) is listed in `requirements.txt`.




# How to run

## Hyperparameter Optimization (HPO)

Hyperparameter optimization requires [hyperopt](https://github.com/hyperopt/hyperopt) version `0.2.7`.

Performing HPO with hyperopt requires running a "master" process which maintains a MongoDB database of hyperparameter configurations (known as trials) that have been tried so far
and decides what configuration to try next.
Then, one or more "workers" reads what configuration to try next, trains a neural network using this configuration, and reports the validation score back to the "master".

For more details, see documentation on hyperopt (http://hyperopt.github.io/hyperopt/) and [this page](http://hyperopt.github.io/hyperopt/scaleout/mongodb/) in particular.
In this project, wrapper scripts have been written to aid in launching the master and workers, explained next.

### Master

Master process is started as follows. Square brackets (`[--abc def]`) denote optional arguments, angle brackets (`<abc>`) are required.

```bash
python HPO_start_master.py <experiment_name> <max_evals> [--DB_host 127.0.0.1] [--DB_port 27017] [--out_dir .] --which_cmd_space <which_cmd_space>
```

For example, to run HPO on the $D_3$ split, with ensembling enabled, as reported in the paper in Figure 3, panel E, labelled *ensembles-HPO* $D_3$ run the following command:

```bash
python HPO_start_master.py "name_for_this_experiment" 2000 --out_dir=/home/ubuntu/ --which_cmd_space=ensembles-HPO_D3_split
```


### Workers

Once the "master" is running. Run one or more workers. Workers must have access to the database on "master".
The worker process is started as follows:

```bash
	python HPO_hyperopt_mongo_worker.py <experiment_name> [--DB_host 127.0.0.1] [--DB_port 27017] [--n_jobs 9999999] [--timeout_hours None]
```

We used SSH port forwarding between an Ubuntu server and compute cluster worker nodes, 
but in this example we assume a worker running on the same node as the master:

```bash
#!/bin/bash
experiment=name_for_this_experiment
LOCALPORT=27017
python -u HPO_hyperopt_mongo_worker.py $experiment "127.0.0.1" $LOCALPORT --timeout_hours 11
```

This process runs for 11 hours, then quits after completing the next trial.
The `-u` flag to python "forces the stdout and stderr streams to be unbuffered" which forces outputs to the screen right away; this is useful when trying to debug problems.

## Neural Network Training

`train_neural_network.py` is the main script used to train the neural networks.
It takes a number of command-line arguments which cover program settings and neural network hyperparameters.
Command-line arguments can be specified on the command line *or* by setting them in a configuration file 
and passing that configuration file to the script using the `--cmd_args_fpath` argument.
For arguments that exist both in the configuration file *and* in the command line, the command line takes precedence.
Here are two examples of training an ensemble of 40:

### 1. Random split

The following command is repeated 50 times produces the boxplot in Figure 3, panel C labelled $CNN_B^{ensemble}$. 
This trains 1 ensemble of 40 models:

```bash
python train_neural_network.py main --resultsDir "C:\my_results" --m "CNNBensemble" --run_baseline --n_in_parallel 4 --n_gpu 1  --n_training_runs 40 --fold_spec "{'type': 'rand_split'}"
```

Set `n_in_parallel` to the number of models to train simultaneously (2 or 3 is typical for the average workstation) and `n_gpu` to 0 to disable the GPU and 1 to enable all available GPUs.

### 2. $D3$ split with ensembling

This command shows how to train using the best-found hyperparameters from HPO using *ensembles-HPO* & $D_3$.
This trains 1 ensemble of 40 models:

```bash
python train_neural_network.py main --resultsDir "C:\my_results" --m "BEST_ensembles-HPO_D3_split" --cmd_args_fpath "./best_models_from_HPO/ensembles-HPO_D3_split.pyon" --n_in_parallel 3 --n_gpu 1
```

In the paper we report the distribution of RMSE scores by repeating the above training 50 times. This produces 50 ensembles, each with its own RMSE on the test set, as reported in the boxplot in Figure 3, panel E labelled *ensembles-HPO* $D_3$.

### Test-set RMSE
`train_neural_network.py` saves the final predictions for each sample in the dataset to the file `best_model_predictions.pkl`.
The file contains a dictionary containing a pandas DataFrame named `df`. In it, column `DM` is the actual DM value, column `DM_pred` is the ensemble's prediction of DM, and column `set` names which set each sample is in.

Test-set RMSE is obtained by comparing test-set predictions to test-set ground truth.


### Results in Figure 3, panel E

The script `reproduce_Figure_3_panel_E.sh` gives the commands used to produce Figure 3, panel E.

Figure 3, panel E, is a boxplot giving distributions over RMSE for the final models (with ensembling in both HPO and final evaluation).
Five models are evaluated, each with a different set of hyperparameter configurations.
Hyperparameter configurations were obtained by HPO on five different choices of validation set and calibration set; the test set is not used in HPO. 
Specifically, the validation sets for each "split" are:

- $D_{rand}$: a random as in previous work
- $D_{2017}$: the latest harvest season (2017)
- $D_1$: first 33% (sorted by date)
- $D_2$: middle 33% (sorted by date)
- $D_3$: last 33% (sorted by date)

The best hyperparameter configurations from each "split"
are saved in configuration files in `best_models_from_HPO`
in `*.pyon` format (these are Python dictionaries written in native Python code).
These are then evaluated by training an ensemble on the original split ($D_{rand}$);
in the config files, this is achieved by setting

```python
'fold_spec': {'type': 'rand_split'}
```

Training is repeated 50 times to produce a distribution of RMSE scores.

For more details, please see the paper.