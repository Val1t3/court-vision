#!/bin/bash

# real distance for all video
length=(15.0 16.4 26.16 15.0 28.0 37.56 20.52 43.0 19.1 17.09 26.6)

# execute sle test for all video
for i in {1..11}; do
    echo "eval_${i} ${length[$((i-1))]}"
    python3 sle.py --name "eval_${i}" --length "${length[$((i-1))]}"
    echo "# Finished processing distances eval_{i} #"
done
