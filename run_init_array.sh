conda activate base_env

python initialize.py $@ --model_name flant5-xl --init_approach greedy --prev_phrase '' --array_job_id $SLURM_ARRAY_TASK_ID


