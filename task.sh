#!/bin/bash

# Define seeds and datasets
seeds=(33)
datasets=("Citeseer")
filename='Citeseer_h2gcn.csv'
model='h2gcn'
attack='metattack'

mkdir -p logs

for seed in "${seeds[@]}"; do
  for dataset in "${datasets[@]}"; do
    echo "Running dataset=$dataset, seed=$seed"
    python attack.py --seed "$seed" --dataset "$dataset" --output "$filename" --attack "$attack" --model "$model" >> logs/sat_aug31242.txt
  done
done

