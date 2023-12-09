#!/bin/bash
#$ -S /bin/bash

source ~/rds/hpc-work/Projects/venv2/bin/activate

python train_attack.py $@ --attack_method greedy --prev_phrase 'resuggest concatenation relation ending relationally'
