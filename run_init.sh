#!/bin/bash
#$ -S /bin/bash

source ~/rds/hpc-work/Projects/venv2/bin/activate

python initialize.py $@ --init_approach greedy --prev_phrase 'resuggest concatenation relation'
