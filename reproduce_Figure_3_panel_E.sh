#!/bin/bash

splits=(
  "ensembles-HPO_Drand_split"
  "ensembles-HPO_D2017_split"
  "ensembles-HPO_D1_split"
  "ensembles-HPO_D2_split"
  "ensembles-HPO_D3_split"
)
for split in "${splits[@]}"; do
  for i in {1..50}
  do
    echo RUNNING: $split, $i
    python train_neural_network.py main --resultsDir /home/ubuntu/results/ --m "${split}_${i}" --cmd_args_fpath "./best_models_from_HPO/${split}.pyon" --n_in_parallel 4 --n_gpu 1 --seed_start ${i}00 --n_training_runs 40 --out_dir_naming AUTO
  done
done
