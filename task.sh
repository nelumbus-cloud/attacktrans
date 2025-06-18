#!/bin/bash
dataset="Amazon-ratings"
seed=11
DIR="logs"
if [ ! -d "$DIR" ]; then
  mkdir "$DIR"
fi

for i in {1..5}
do
    timestamp=$(date +%Y%m%d_%H%M%S)
    output_file="logs/${dataset}_${seed}_${timestamp}.txt"

    echo "Running attack with seed $seed..."
    python attack.py --seed $seed > "$output_file"
    echo "Finished attack with seed $seed!"

    seed=$((seed + 10))  # increment seed

done
