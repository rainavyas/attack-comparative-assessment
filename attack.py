'''
    Evaluate attack
'''

import sys
import os
import torch
import numpy as np

from src.tools.tools import get_default_device, set_seeds
from src.tools.args import core_args, attack_args
from src.tools.saving import base_path_creator, attack_base_path_creator_eval
from src.data.load_data import load_data
from src.models import load_model
from src.attacker.selector import select_eval_attacker

if __name__ == "__main__":

    # get command line arguments
    core_args, c = core_args()
    attack_args, a = attack_args()

    print(core_args)
    print(attack_args)
    
    set_seeds(core_args.seed)
    base_path = base_path_creator(core_args)
    attack_base_path = attack_base_path_creator_eval(attack_args, base_path)

    # Save the command run
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/attack.cmd', 'a') as f:
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
    model = load_model(core_args, device=device, assessment=core_args.assessment)

    # load attacker for evaluation
    attacker = select_eval_attacker(attack_args, core_args, model)

    # evaluate

    # 1) No attack
    if not attack_args.not_none:
        print('No attack')
        result = attacker.eval_uni_attack(test_data, adv_phrase='', cache_dir=base_path, force_run=attack_args.force_run)
        print(result)
        print()

    # 2) Attack i
    print('Attack i')
    result = attacker.eval_uni_attack(test_data, adv_phrase=attacker.adv_phrase, cache_dir=attack_base_path, force_run=attack_args.force_run)
    print(result)
    print()

    

