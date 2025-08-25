#!/bin/bash

# execute python script for all videos
# for i in {1..11}; do
#     python3 dist_calc.py --csv "../data/courtvision-dataset/eval_${i}_tracks.csv" --img-points ../data/data/points_cropped_schema.json --court-points ../data/data/eval_points.json --video "../data/assets/eval_${i}.mov" --output "../data/courtvision-dataset/eval_${i}_dist.csv"
#     echo "Finished processing eval_${i}_tracks.csv"
# done

# calculate distances for all video
for i in {4..11}; do
    python3 distance_calculation.py --name "eval_${i}"
    echo "# Finished processing distances eval_${i} #"
done
