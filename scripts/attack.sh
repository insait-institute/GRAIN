#!/bin/bash
array=( $@ )
len=${#array[@]}
last_args=${array[@]:1:$len}

python ./src/attack.py --dataset $1 $last_args $last_args
