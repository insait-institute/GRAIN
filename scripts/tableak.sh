#!/bin/bash
array=( $@ )
len=${#array[@]}
last_args=${array[@]:1:$len}

python ./src/tableak.py --dataset $1 $last_args

