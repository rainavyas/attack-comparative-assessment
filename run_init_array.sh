#!/bin/bash

source ~/.bashrc
conda activate base_env
export OMP_NUM_THREADS=30

python train_attack.py $@ --model_name flant5-xl --attack_method greedy --prev_phrase 'uncontradictory' --array_job_id $SLURM_ARRAY_TASK_ID


