#!/bin/bash

# execute python script for all videos
for i in {1..11}; do
	python3 main.py "../data/assets/eval_${i}.mov" --out-video ../data/courtvision-dataset/eval_${i}_tracked.mp4 --out-csv ../data/courtvision-dataset/eval_${i}_tracks.csv
    echo "Finished processing eval_${i}.mov"
done
