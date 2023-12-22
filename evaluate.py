'''
Generic functions to process saved outputs for presenting results
'''

import numpy as np
import os
import sys
import torch

from src.tools.tools import get_default_device, set_seeds
from src.tools.args import core_args, attack_args, process_args

from src.data.load_data import load_data
from src.models import load_model
from src.attacker.selector import select_eval_attacker
from src.tools.saving import base_path_creator


if __name__ == "__main__":

    # get command line arguments
    core_args, c = core_args()
    attack_args, a = attack_args()
    process_args, p = process_args()

    print(core_args)
    print(attack_args)

    base_path = base_path_creator(core_args)

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/evaluate.cmd', 'a') as f:
        f.write(' '.join(sys.argv)+'\n')

    # Get the device
    if core_args.force_cpu:
        device = torch.device('cpu')
    else:
        device = get_default_device(core_args.gpu_id)
    print(device)
    
    # Load the data
    train_data, test_data = load_data(core_args)
    if core_args.eval_train:
        test_data = train_data

    # Load the model, tokenizer
    model = load_model(model_name=core_args.model_name, device=device, assessment=core_args.assessment)

    # load attacker for evaluation
    attacker = select_eval_attacker(attack_args, core_args, model)

    result = attacker.spearman_rank_performance(test_data, cache_dir=base_path, force_run=attack_args.force_run)
    print(result)